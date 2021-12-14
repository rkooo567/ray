// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/stats/metric_defs.h"

namespace ray {

namespace stats {

/// The definitions of metrics that you can use everywhere.
///
/// There are 4 types of metric:
///   Histogram: Histogram distribution of metric points.
///   Gauge: Keeps the last recorded value, drops everything before.
///   Count: The count of the number of metric points.
///   Sum: A sum up of the metric points.
///
/// You can follow these examples to define your metrics.

/// NOTE: When adding a new metric, add the metric name to the _METRICS list in
/// python/ray/tests/test_metrics_agent.py to ensure that its existence is tested.

/// Event stats
DEFINE_stats(operation_count, "operation count", ("Method"), (), ray::stats::GAUGE);
DEFINE_stats(operation_run_time_ms, "operation execution time", ("Method"), (),
             ray::stats::GAUGE);
DEFINE_stats(operation_queue_time_ms, "operation queuing time", ("Method"), (),
             ray::stats::GAUGE);
DEFINE_stats(operation_active_count, "activate operation number", ("Method"), (),
             ray::stats::GAUGE);

/// GRPC server
DEFINE_stats(grpc_server_req_process_time_ms, "Request latency in grpc server",
             ("Method"), (), ray::stats::GAUGE);
DEFINE_stats(grpc_server_req_new, "New request number in grpc server", ("Method"), (),
             ray::stats::COUNT);
DEFINE_stats(grpc_server_req_handling, "Request number are handling in grpc server",
             ("Method"), (), ray::stats::COUNT);
DEFINE_stats(grpc_server_req_finished, "Finished request number in grpc server",
             ("Method"), (), ray::stats::COUNT);

/// Object Manager.
DEFINE_stats(object_manager_chunks_received_count,
             "Number object chunks received broken per type {Total, FailedTotal, "
             "FailedCancelled, FailedPlasmaFull}.",
             ("Type"), (), ray::stats::GAUGE);

/// Pull Manager
DEFINE_stats(pull_manager_bytes_usage,
             "The total number of bytes usage per type {Available, BeingPulled, Pinned}",
             ("Type"), (), ray::stats::GAUGE);
DEFINE_stats(pull_manager_requested_bundles,
             "Number of requested bundles per type {Get, Wait, TaskArgs}.", ("Type"), (),
             ray::stats::GAUGE);
DEFINE_stats(pull_manager_requests,
             "Number of pull requests per type {Queued, Active, Pinned, ActiveBundles}.",
             ("Type"), (), ray::stats::GAUGE);
DEFINE_stats(pull_manager_retries_count, "Number of cumulative pull retries.", (), (),
             ray::stats::GAUGE);

/// Push Manager
DEFINE_stats(push_manager_push_count,
             "Number of object push requests per type {InFlight}.", ("Type"), (),
             ray::stats::GAUGE);
DEFINE_stats(push_manager_chunks_count,
             "Number of object chunks transfer per type {InFlight, Remaining}.", ("Type"),
             (), ray::stats::GAUGE);

/// Scheduler
DEFINE_stats(scheduler_tasks_count,
             "Number of tasks waiting for scheduling per state {Cancelled, Executing, "
             "Waiting, Dispatched, Received}.",
             ("State"), (), ray::stats::GAUGE);
DEFINE_stats(
    scheduler_pending_tasks_count,
    "Number of tasks waiting for scheduling per reason {Infeasible, WaitingForResources, "
    "WaitingForPlasmaMemory, WaitingForRemoteResources, WaitingForWorkers}.",
    ("Reason"), (), ray::stats::GAUGE);
DEFINE_stats(
    scheduler_failed_worker_startup_count,
    "Number of tasks that fail to be scheduled because workers were not "
    "available. Labels per reason {JobConfigMissing, RegistrationTimedOut, RateLimited}",
    ("Reason"), (), ray::stats::GAUGE);

/// Local Object Manager
DEFINE_stats(spill_manager_objects_count,
             "Number of local objects per state {Pinned, PendingRestore, PendingSpill}.",
             ("State"), (), ray::stats::GAUGE);
DEFINE_stats(spill_manager_objects_bytes, "Byte size of local objects per state.",
             ("State"), (), ray::stats::GAUGE);
DEFINE_stats(spill_manager_cumulative_request_count,
             "Number of {spill, restore} requests.", ("Type"), (), ray::stats::GAUGE);
DEFINE_stats(spill_manager_throughput_mb,
             "The throughput of {spill, restore} requests in MB.", ("Type"), (),
             ray::stats::GAUGE);

///
/// Plasma Store Metrics
///

/// Object Lifecycle Manager.
DEFINE_stats(plasma_num_local_objects_by_state,
             "The number of objects per state. E.g., Spillable, In use, Evictable.",
             ("State"), (), ray::stats::GAUGE);
DEFINE_stats(plasma_num_local_bytes_by_state, "The number of objects per state in bytes.",
             ("State"), (), ray::stats::GAUGE);
DEFINE_stats(
    plasma_num_local_objects_by_type,
    "The number of objects per type. E.g., Primary copy, Transferred, ErrorObject.",
    ("CreationType"), (), ray::stats::GAUGE);
DEFINE_stats(plasma_num_local_bytes_by_type, "The number of objects per type in bytes.",
             ("CreationType"), (), ray::stats::GAUGE);

/// Plasma Store
DEFINE_stats(num_pending_creation_requests,
             "The number of pending object creation requests in the queue.", (), (),
             ray::stats::GAUGE);
DEFINE_stats(num_pending_creation_bytes,
             "The number of pending object creation requests in bytes.", (), (),
             ray::stats::GAUGE);

/// GCS Resource Manager
DEFINE_stats(new_resource_creation_latency_ms,
             "Time to persist newly created resources to Redis.", (),
             ({0.1, 1, 10, 100, 1000, 10000}, ), ray::stats::HISTOGRAM);

/// Placement Group
// The end to end placement group creation latency.
// The time from placement group creation request has received
// <-> Placement group creation succeeds (meaning all resources
// are committed to nodes and available).
DEFINE_stats(placement_group_creation_latency_ms,
             "end to end latency of placement group creation", (),
             ({0.1, 1, 10, 100, 1000, 10000}, ), ray::stats::HISTOGRAM);
// The time from placement group scheduling has started
// <-> Placement group creation succeeds.
DEFINE_stats(placement_group_scheduling_latency_ms,
             "scheduling latency of placement groups", (),
             ({0.1, 1, 10, 100, 1000, 10000}, ), ray::stats::HISTOGRAM);
DEFINE_stats(pending_placement_group, "Number of total pending placement groups", (), (),
             ray::stats::GAUGE);
DEFINE_stats(registered_placement_group, "Number of total registered placement groups",
             (), (), ray::stats::GAUGE);
DEFINE_stats(infeasible_placement_group, "Number of total infeasible placement groups",
             (), (), ray::stats::GAUGE);

}  // namespace stats

}  // namespace ray
