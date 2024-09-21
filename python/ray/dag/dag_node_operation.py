from functools import total_ordering
from enum import Enum
from typing import Optional, Tuple, List, Dict
import graphviz
import ray
import heapq
from collections import defaultdict


class _DAGNodeOperationType(Enum):
    """
    There are three types of operations that a DAG node can perform:
    1. READ: Read from an input channel.
    2. COMPUTE: Execute the method corresponding to the node.
    3. WRITE: Write to an output channel.
    """

    READ = "READ"
    COMPUTE = "COMPUTE"
    WRITE = "WRITE"

    def __str__(self):
        if self == _DAGNodeOperationType.READ:
            return "R"
        elif self == _DAGNodeOperationType.COMPUTE:
            return "C"
        elif self == _DAGNodeOperationType.WRITE:
            return "W"
        assert False, f"Unknown operation type: {self}"


class _DAGNodeOperation:
    def __init__(
        self,
        exec_task_idx: int,
        operation_type: _DAGNodeOperationType,
        method_name: Optional[str] = None,
    ):
        """
        Args:
            exec_task_idx: The index of the task that this operation belongs to
                in the actor's ExecutableTask list. The index is not the same
                as bind_index because there may be more tasks bound to an actor
                than tasks that appear in the current compiled DAG.
            operation_type: The type of operation to perform.
            method_name: The name of the method that this operation originates
                from. This is only for debugging purposes.
        """
        self.exec_task_idx = exec_task_idx
        self.type = operation_type
        self.method_name = method_name

    def next_operation(self):
        if self.type == _DAGNodeOperationType.READ:
            return _DAGNodeOperation(
                self.exec_task_idx, _DAGNodeOperationType.COMPUTE, self.method_name
            )
        elif self.type == _DAGNodeOperationType.COMPUTE:
            return _DAGNodeOperation(
                self.exec_task_idx, _DAGNodeOperationType.WRITE, self.method_name
            )
        else:
            raise ValueError(
                "Cannot only get next operation for READ or COMPUTE type, "
                f"{self.type} is provided."
            )

    def __repr__(self):
        return f"([{self.exec_task_idx}] {self.method_name} {self.type})"
        # return f"(Task idx: {self.exec_task_idx}, Type: {self.type})"

    def __str__(self):
        return f"([{self.exec_task_idx}] {self.method_name} {self.type})"

    def __hash__(self):
        return hash((self.exec_task_idx, self.type))

    def __eq__(self, other):
        # An operation is uniquely identified by its `exec_task_idx` and type.
        # `func_name` is only for debugging purposes.
        return self.exec_task_idx == other.exec_task_idx and self.type == other.type


@total_ordering
class _DAGOperationGraphNode:
    def __init__(
        self,
        operation: _DAGNodeOperation,
        task_idx: int,
        actor_handle: "ray.actor.ActorHandle",
        requires_nccl: bool,
    ):
        """
        _DAGOperationGraphNode represents a node in the DAG operation graph.
        It contains information about the node's in-degree, out-degree, edges,
        and the operation it performs.

        Args:
            operation: The operation that this node performs. The operation
                can be a READ, COMPUTE, or WRITE operation.
            task_idx: A unique index which can be used to index into
                `CompiledDAG.idx_to_task` to get the corresponding task.
            actor_handle: The actor handle to which this operation belongs.
            requires_nccl: Whether this operation requires NCCL.
        """
        self.operation = operation
        self.task_idx = task_idx
        self.actor_handle = actor_handle
        self.requires_nccl = requires_nccl
        # The in_edges and out_edges are sets of tuples. Each tuple contains
        # an integer `task_idx`, which can be used to index into `idx_to_task`
        # to get the corresponding task, and a `_DAGNodeOperationType`, which can
        # be READ, COMPUTE, or WRITE.
        self.in_edges: Dict[Tuple[int, _DAGNodeOperationType]] = {}
        self.out_edges: Dict[Tuple[int, _DAGNodeOperationType]] = {}

    @property
    def in_degree(self) -> int:
        return len(self.in_edges)

    def __lt__(self, other: "_DAGOperationGraphNode"):
        """
        This function defines the order of the nodes in the priority queue used in
        `_select_next_nodes`. The priority queue is a min-heap, so the node with
        higher priority is considered "less than" the other node.
        """
        # If two nodes belong to the same actor, select the one with
        # the smaller `exec_task_idx`.
        if self.actor_handle == other.actor_handle:
            return self.operation.exec_task_idx < other.operation.exec_task_idx
        # If two nodes belong to different actors and one of them is an NCCL
        # write node, select the one that is not an NCCL write node.
        is_nccl_write = (
            self.operation.type == _DAGNodeOperationType.WRITE and self.requires_nccl
        )
        other_is_nccl_write = (
            other.operation.type == _DAGNodeOperationType.WRITE and other.requires_nccl
        )
        if is_nccl_write != other_is_nccl_write:
            return not is_nccl_write
        # If two nodes belong to different actors and both are either NCCL write
        # nodes or neither are NCCL write nodes, select the one with the smaller
        # `exec_task_idx`. If they have the same `exec_task_idx`, select the one
        # with the smaller `task_idx`.
        if self.operation.exec_task_idx != other.operation.exec_task_idx:
            return self.operation.exec_task_idx < other.operation.exec_task_idx
        return self.task_idx < other.task_idx

    def __eq__(self, other: "_DAGOperationGraphNode"):
        """
        Two operations are equal only when they have the same `exec_task_idx` and `type`
        and belong to the same actor.
        """
        return (
            self.actor_handle == other.actor_handle
            and self.operation.exec_task_idx == other.operation.exec_task_idx
            and self.operation.type == other.operation.type
        )

    def __hash__(self):
        """
        An operation is uniquely identified by its `task_idx` and type.
        """
        return hash((self.operation, self.task_idx))

    def __str__(self):
        class_name = (
            self.actor_handle._ray_actor_creation_function_descriptor.class_name
        )
        actor_id = self.actor_handle._actor_id.hex()
        return (
            class_name
            + "_"
            + actor_id[:4]
            + f" [{self.operation.exec_task_idx}] "
            + f"{self.operation.method_name} {self.operation.type}"
        )

    def _get_actor_id(self):
        return self.actor_handle._ray_actor_id.hex()


def _add_edge(
    from_node: _DAGOperationGraphNode, to_node: _DAGOperationGraphNode, label=""
):
    """
    Add an edge from `from_node` to `to_node`. An edge is a tuple of
    the operation's `task_idx` and type.
    """
    from_node.out_edges[(to_node.task_idx, to_node.operation.type)] = label
    to_node.in_edges[(from_node.task_idx, from_node.operation.type)] = label


def _select_next_nodes(
    actor_to_candidates: Dict["ray._raylet.ActorID", List[_DAGOperationGraphNode]],
    graph: Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]],
):
    """
    This function selects the next nodes for topological sort to generate execution
    schedule. If there are multiple candidate _DAGOperationGraphNodes, select the node
    with the top priority based on the following rules:

    #1  If two candidate nodes belong to the same actor, select the one with
        the smaller `exec_task_idx`.

    #2  If two candidate nodes belong to different actors and both are either NCCL
        write nodes or neither are NCCL write nodes, select the one with the smaller
        `exec_task_idx`. If they have the same `exec_task_idx`, select the one with the
        smaller `task_idx`.

    #3  If two candidate nodes belong to different actors and one of them is an NCCL
        write node, select the one that is not an NCCL write node.

    For the implementation details, we maintain a priority queue for each actor,
    where the head of the priority queue is the node with the smallest `exec_task_idx`.

    If the selected node is an NCCL write node, select all its immediately downstream
    nodes, which are NCCL read nodes, regardless of whether the downstream nodes are
    heads of their own priority queues. In that case, this function only removes the
    NCCL write node, which is also the head of a priority queue. Other nodes will be
    removed in the following iterations. The NCCL read nodes will be returned even
    though they should not yet be in the candidate list.

    Args:
        actor_to_candidates: A dictionary mapping an actor id to a list of
            candidate nodes. The list is maintained as a priority queue, so
            the head of the queue, i.e., `candidates[0]`, is the node with
            the smallest `bind_index`.
        graph: A dictionary mapping the index of a task to a dictionary of its
            _DAGOperationGraphNodes for different operations.

    Returns:
        A list of _DAGOperationGraphNodes to be placed into the corresponding
        execution schedules.
    """
    top_priority_node = None
    next_nodes: List[_DAGOperationGraphNode] = []
    for _, candidates in actor_to_candidates.items():
        if len(candidates) == 0:
            continue
        if top_priority_node is None or candidates[0] < top_priority_node:
            top_priority_node = candidates[0]
    assert top_priority_node is not None
    next_nodes.append(
        heapq.heappop(actor_to_candidates[top_priority_node.actor_handle._actor_id])
    )

    if not (
        top_priority_node.operation.type == _DAGNodeOperationType.WRITE
        and top_priority_node.requires_nccl
    ):
        assert len(next_nodes) == 1
        return next_nodes

    # An NCCL write node is picked. NCCL is a blocking operation, so we need to pick all
    # the corresponding NCCL read nodes to avoid a deadlock.
    for downstream_node_metadata in top_priority_node.out_edges:
        task_idx, op_type = downstream_node_metadata[0], downstream_node_metadata[1]
        downstream_node = graph[task_idx][op_type]
        assert downstream_node.operation.type == _DAGNodeOperationType.READ
        next_nodes.append(downstream_node)
    assert len(next_nodes) == 1 + len(top_priority_node.out_edges)
    return next_nodes


def _visualize_graph(
    graph: Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]]
):
    dot = graphviz.Digraph(comment="DAG")

    actor_to_nodes = defaultdict(list)

    # Add nodes and edges to the graph
    for task_idx, dict in graph.items():
        for node in dict.values():
            node_label = str(node)
            dot.node(node_label, node_label)

            actor_to_nodes[node._get_actor_id()].append(node)

            # # Add in_edges
            # for in_edge, label in node.in_edges.items():
            #     in_task_idx, in_op_type = in_edge
            #     in_node = graph[in_task_idx][in_op_type]
            #     dot.edge(str(in_node), str(node), label="")

            # Add out_edges
            for out_edge, label in node.out_edges.items():
                out_task_idx, out_op_type = out_edge
                out_node = graph[out_task_idx][out_op_type]
                color = "blue" if label == "nccl" else "black"
                dot.edge(node_label, str(out_node), label=label, color=color)

    for actor_id, nodes in actor_to_nodes.items():
        with dot.subgraph(name=f"cluster_{actor_id}") as subgraph:
            subgraph.attr(rank=nodes[0]._get_actor_id())
            for node in nodes:
                subgraph.node(str(node), str(node))

    # Render the graph to a file or display it
    dot.render("dag_graph", format="png", view=True)


def _build_dag_node_operation_graph(
    idx_to_task: Dict[int, "ray.dag.compiled_dag_node.CompiledTask"],
    actor_to_operation_nodes: Dict[
        "ray.actor.ActorHandle", List[List[_DAGOperationGraphNode]]
    ],
) -> Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]]:
    """
    Generate a DAG node operation graph by adding edges based on the
    following rules:

    #1  Add edges from READ to COMPUTE, and from COMPUTE to WRITE, which
        belong to the same task.
    #2  Add an edge from COMPUTE with bind_index i to COMPUTE with bind_index
        i+1 if they belong to the same actor.
    #3  Add an edge from WRITE of the writer task to READ of the reader task.

    This is the step one of building an execution schedule for each actor.

    Args:
        idx_to_task: A dictionary that maps the `task_idx` to the `CompiledTask`.
            `CompiledTask` contains information about a DAGNode and its downstream
            nodes.

        actor_to_operation_nodes: A dictionary that maps an actor handle to
            a list of lists of _DAGOperationGraphNode. For the same actor, the
            index of the outer list corresponds to the index of the ExecutableTask
            in the list of `executable_tasks` in `actor_to_executable_tasks`. In
            the inner list, the order of operations is READ, COMPUTE, and WRITE.

    Returns:
        A graph where each node is a _DAGOperationGraphNode. The key is `task_idx`,
        the index to retrieve its task from `idx_to_task`, and the value is a
        dictionary that maps the _DAGNodeOperationType (READ, COMPUTE, or WRITE)
        to the corresponding _DAGOperationGraphNode
    """
    assert idx_to_task
    graph: Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]] = {}

    for _, operation_nodes_list in actor_to_operation_nodes.items():
        prev_compute_node = None
        for operation_nodes in operation_nodes_list:
            task_idx = operation_nodes[0].task_idx
            read_node, compute_node, write_node = (
                operation_nodes[0],
                operation_nodes[1],
                operation_nodes[2],
            )
            # Add edges from READ to COMPUTE, and from COMPUTE to WRITE, which
            # belong to the same task.
            _add_edge(read_node, compute_node)
            _add_edge(compute_node, write_node)
            # Add an edge from COMPUTE with `bind_index` i to COMPUTE with
            # `bind_index` i+1 if they belong to the same actor.
            if prev_compute_node is not None:
                _add_edge(prev_compute_node, compute_node, "next")
            prev_compute_node = compute_node
            assert task_idx not in graph
            graph[task_idx] = {
                _DAGNodeOperationType.READ: read_node,
                _DAGNodeOperationType.COMPUTE: compute_node,
                _DAGNodeOperationType.WRITE: write_node,
            }

    # Import `ray.dag` here to avoid circular import.
    from ray.dag import ClassMethodNode, MultiOutputNode

    # Add an edge from WRITE of the writer task to READ of the reader task.
    for task_idx, task in idx_to_task.items():
        if (
            isinstance(task.dag_node, ClassMethodNode)
            and task.dag_node.is_class_method_output
        ):
            # TODO(wxdeng): Handle the case where the task is a class method output.
            continue
        if not isinstance(task.dag_node, ClassMethodNode):
            # The graph is used to generate an execution schedule for each actor.
            # The edge from the InputNode has no impact on the final execution
            # schedule.
            continue
        for downstream_task_idx in task.downstream_task_idxs:
            downstream_dag_node = idx_to_task[downstream_task_idx].dag_node
            if (
                isinstance(downstream_dag_node, ClassMethodNode)
                and downstream_dag_node.is_class_method_output
            ):
                # TODO(wxdeng): Handle the case where the downstream task is
                # a class method output.
                continue
            if isinstance(downstream_dag_node, MultiOutputNode):
                continue
            _add_edge(
                graph[task_idx][_DAGNodeOperationType.WRITE],
                graph[downstream_task_idx][_DAGNodeOperationType.READ],
                "nccl"
                if graph[task_idx][_DAGNodeOperationType.WRITE].requires_nccl
                else "shm",
            )
    # _visualize_graph(graph)
    return graph


def _node_repr(node: _DAGOperationGraphNode, idx: int, optimized_index):
    return str(node) + f" {idx},{optimized_index}"


def _visualize_graph_ordered(
    actor_to_execution_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
    actor_to_optimized_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
    graph: Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]],
):
    dot = graphviz.Digraph(comment="DAG")
    node_to_node_repr = {}

    for actor, execution_nodes in actor_to_execution_nodes.items():
        optimized_nodes = actor_to_optimized_nodes[actor]
        node_to_optimized_index = {node: i for i, node in enumerate(optimized_nodes)}

        with dot.subgraph(
            name=f"cluster_{execution_nodes[0]._get_actor_id()}"
        ) as subgraph:
            subgraph.attr(rank=execution_nodes[0]._get_actor_id())
            for i, node in enumerate(execution_nodes):
                optimized_index = node_to_optimized_index.get(node)
                node_repr = _node_repr(node, i, optimized_index)
                color = "red" if optimized_index != i else "black"
                subgraph.node(node_repr, node_repr, color=color)
                node_to_node_repr[node] = node_repr

    for actor, execution_nodes in actor_to_execution_nodes.items():
        for i, node in enumerate(execution_nodes):
            node_repr = node_to_node_repr[node]
            for out_edge, label in node.out_edges.items():
                out_task_idx, out_op_type = out_edge
                out_node = graph[out_task_idx][out_op_type]
                out_node_repr = node_to_node_repr[out_node]
                color = "blue" if label == "nccl" else "black"
                dot.edge(node_repr, out_node_repr, label=label, color=color)

    # Render the graph to a file or display it
    dot.render("dag_schedule", format="png", view=True)


def _generate_actor_to_execution_schedule(
    graph: Dict[int, Dict[_DAGNodeOperationType, _DAGOperationGraphNode]]
) -> Dict["ray.actor.ActorHandle", List[_DAGNodeOperation]]:
    """
    Generate an execution schedule for each actor. The schedule is a list of
    operations to be executed. The function uses a topological sort algorithm
    to generate the schedule.

    Args:
        graph: A graph where each node is a _DAGOperationGraphNode. The key is
            `task_idx`, the index to retrieve its task from `idx_to_task`, and
            the value is a dictionary that maps the _DAGNodeOperationType (READ,
            COMPUTE, or WRITE) to the corresponding _DAGOperationGraphNode. It is
            generated by `_build_dag_node_operation_graph`.

    Returns:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the execution schedule which is a list of operations to be executed.
    """

    # Mapping from the actor handle to the execution schedule which is a list
    # of operations to be executed.
    actor_to_execution_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGNodeOperation]
    ] = defaultdict(list)
    actor_to_execution_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ] = defaultdict(list)

    # A dictionary mapping an actor id to a list of candidate nodes. The list
    # is maintained as a priority queue, so the head of the queue, i.e.,
    # `candidates[0]`, is the node with the smallest `bind_index`.
    actor_to_candidates: Dict[
        "ray._raylet.ActorID", List[_DAGOperationGraphNode]
    ] = defaultdict(list)
    for _, node_dict in graph.items():
        for _, node in node_dict.items():
            # A node with a zero in-degree edge means all of its dependencies
            # have been satisfied, including both data and control dependencies.
            # Therefore, it is a candidate for execution.
            if node.in_degree == 0:
                heapq.heappush(actor_to_candidates[node.actor_handle._actor_id], node)

    visited_nodes = set()

    # Use topological sort algorithm to generate the execution schedule. Each iteration
    # pops a candidate node from `actor_to_candidates` and each DAG node consists of
    # three operations: READ, COMPUTE, and WRITE.
    for _ in range(len(graph) * 3):
        # The function `_select_next_nodes` will pop a candidate node from
        # `actor_to_candidates` and return a list of nodes that can be executed
        # in the next step. If multiple nodes are returned, only the NCCL write
        # node is popped in this iteration.
        nodes = _select_next_nodes(actor_to_candidates, graph)
        for node in nodes:
            if node in visited_nodes:
                continue
            actor_to_execution_schedule[node.actor_handle].append(node.operation)
            actor_to_execution_nodes[node.actor_handle].append(node)
            visited_nodes.add(node)
            for out_node_task_idx, out_node_type in node.out_edges:
                out_node = graph[out_node_task_idx][out_node_type]
                out_node.in_edges.pop((node.task_idx, node.operation.type))
                if out_node.in_degree == 0:
                    heapq.heappush(
                        actor_to_candidates[out_node.actor_handle._actor_id],
                        out_node,
                    )
    for _, candidates in actor_to_candidates.items():
        assert len(candidates) == 0
    for actor_handle, execution_schedule in actor_to_execution_schedule.items():
        print(f"Actor {actor_handle._ray_actor_id} schedule: {execution_schedule}")
    # _visualize_graph_ordered(actor_to_nodes, graph)
    return actor_to_execution_schedule, actor_to_execution_nodes


def _optimize_execution_schedule(
    actor_to_execution_schedule: Dict["ray.actor.ActorHandle", List[_DAGNodeOperation]],
    actor_to_execution_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
    out_of_order_limit: int = 1,
):
    """
    Optimize the execution schedule by overlapping computation and communication.

    Args:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the execution schedule which is a list of operations to be executed.
        out_of_order_limit: The maximum number of out-of-order `receive` operations
            allowed.
    """
    if out_of_order_limit == 0:
        return actor_to_execution_schedule, actor_to_execution_nodes

    actor_to_optimized_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGNodeOperation]
    ] = defaultdict(list)
    actor_to_optimized_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ] = defaultdict(list)

    for actor, execution_nodes in actor_to_execution_nodes.items():
        fast = 0
        slow = 0
        optimized_schedule = []
        optimized_nodes = []
        out_of_order_quota = out_of_order_limit + 1
        while slow < len(execution_nodes):
            while out_of_order_quota > 0 and fast < len(execution_nodes):
                if (
                    execution_nodes[fast].operation.type == _DAGNodeOperationType.READ
                    and execution_nodes[fast].requires_nccl
                ):
                    optimized_nodes.append(execution_nodes[fast])
                    optimized_schedule.append(execution_nodes[fast].operation)
                    out_of_order_quota -= 1
                fast += 1
            while not out_of_order_quota or fast >= len(execution_nodes):
                if (
                    execution_nodes[slow].operation.type != _DAGNodeOperationType.READ
                    or not execution_nodes[slow].requires_nccl
                ):
                    optimized_nodes.append(execution_nodes[slow])
                    optimized_schedule.append(execution_nodes[slow].operation)
                    if (
                        execution_nodes[slow].operation.type
                        == _DAGNodeOperationType.WRITE
                        and execution_nodes[slow].requires_nccl
                    ):
                        out_of_order_quota += 1
                slow += 1
        actor_to_optimized_nodes[actor] = optimized_nodes
        actor_to_optimized_schedule[actor] = optimized_schedule
        print(f"Actor {actor._ray_actor_id} optimized schedule:", optimized_schedule)
    return actor_to_optimized_schedule, actor_to_optimized_nodes


def _optimize_execution_schedule_bak(
    actor_to_execution_schedule: Dict["ray.actor.ActorHandle", List[_DAGNodeOperation]],
    actor_to_execution_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
    out_of_order_limit: int = 1,
):
    """
    Optimize the execution schedule by overlapping computation and communication.

    Args:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the execution schedule which is a list of operations to be executed.
        out_of_order_limit: The maximum number of out-of-order `receive` operations
            allowed.
    """
    # TODO: analyze the DAG and turn off overlap optimization when it is
    # not supported (yet). For example, currently if a channel requires
    # both NCCL and shared memory transport, overlap optimization cannot
    # be applied.
    if out_of_order_limit == 0:
        return actor_to_execution_schedule, actor_to_execution_nodes

    actor_to_optimized_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGNodeOperation]
    ] = defaultdict(list)
    actor_to_optimized_nodes: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ] = defaultdict(list)
    for actor, execution_nodes in actor_to_execution_nodes.items():
        read_queue = []
        other_queue = []
        optimized_schedule = []
        optimized_nodes = []
        for node in execution_nodes:
            if node.operation.type == _DAGNodeOperationType.READ:
                read_queue.append(node)
            else:
                other_queue.append(node)
        out_of_order_quota = out_of_order_limit + 1
        while other_queue:
            other_node = other_queue[0]
            if read_queue:
                if out_of_order_quota > 0:
                    picked = read_queue.pop(0)
                    optimized_nodes.append(picked)
                    optimized_schedule.append(picked.operation)
                    out_of_order_quota -= 1
                else:
                    picked = other_queue.pop(0)
                    optimized_nodes.append(picked)
                    optimized_schedule.append(picked.operation)
                    if other_node.operation.type == _DAGNodeOperationType.WRITE:
                        out_of_order_quota += 1
            else:
                picked = other_queue.pop(0)
                optimized_nodes.append(picked)
                optimized_schedule.append(picked.operation)
        actor_to_optimized_nodes[actor] = optimized_nodes
        actor_to_optimized_schedule[actor] = optimized_schedule
        print(f"Actor {actor._ray_actor_id} optimized schedule:", optimized_schedule)
    return actor_to_optimized_schedule, actor_to_optimized_nodes
