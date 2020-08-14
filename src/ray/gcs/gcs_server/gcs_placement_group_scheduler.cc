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

#include "ray/gcs/gcs_server/gcs_placement_group_scheduler.h"

#include "ray/gcs/gcs_server/gcs_placement_group_manager.h"
#include "src/ray/protobuf/gcs.pb.h"

namespace ray {
namespace gcs {

GcsPlacementGroupScheduler::GcsPlacementGroupScheduler(
    boost::asio::io_context &io_context,
    std::shared_ptr<gcs::GcsTableStorage> gcs_table_storage,
    const gcs::GcsNodeManager &gcs_node_manager,
    ReserveResourceClientFactoryFn lease_client_factory)
    : return_timer_(io_context),
      gcs_table_storage_(std::move(gcs_table_storage)),
      gcs_node_manager_(gcs_node_manager),
      lease_client_factory_(std::move(lease_client_factory)) {
  scheduler_strategies_.push_back(std::make_shared<GcsPackStrategy>());
  scheduler_strategies_.push_back(std::make_shared<GcsSpreadStrategy>());
}

/// In this algorithm, we try to pack all the bundles in the node which satisfies the
/// resource requirements and has the least number of bundles.
/// TODO(ffbin): At present, only one node will be scheduled. If one node does not have
/// enough resources, we need to divide bundles to multiple nodes. We will implement
/// it in the next pr.
ScheduleMap GcsPackStrategy::Schedule(
    std::vector<std::shared_ptr<ray::BundleSpecification>> &bundles,
    const std::unique_ptr<ScheduleContext> &context) {
  // Aggregate required resources.
  ResourceSet required_resources;
  for (const auto &bundle : bundles) {
    required_resources.AddResources(bundle->GetRequiredResources());
  }

  // Filter candidate nodes.
  const auto &alive_nodes = context->node_manager_.GetClusterRealtimeResources();
  std::vector<std::pair<int64_t, ClientID>> candidate_nodes;
  for (auto &node : alive_nodes) {
    if (required_resources.IsSubset(*node.second)) {
      candidate_nodes.emplace_back((*context->node_to_bundles_)[node.first], node.first);
    }
  }

  // Select the node with the least number of bundles.
  ScheduleMap schedule_map;
  if (candidate_nodes.empty()) {
    return schedule_map;
  }

  std::sort(
      std::begin(candidate_nodes), std::end(candidate_nodes),
      [](const std::pair<int64_t, ClientID> &left,
         const std::pair<int64_t, ClientID> &right) { return left.first < right.first; });

  for (auto &bundle : bundles) {
    schedule_map[bundle->BundleId()] = candidate_nodes.front().second;
  }
  return schedule_map;
}

/// This is an initial algorithm to respect spread algorithm.
/// In this algorithm, we try to spread all the bundle in different node
/// and don't care the real resource.
ScheduleMap GcsSpreadStrategy::Schedule(
    std::vector<std::shared_ptr<ray::BundleSpecification>> &bundles,
    const std::unique_ptr<ScheduleContext> &context) {
  ScheduleMap schedule_map;
  auto &alive_nodes = context->node_manager_.GetClusterRealtimeResources();
  auto iter = alive_nodes.begin();
  size_t index = 0;
  size_t alive_nodes_size = alive_nodes.size();
  for (; iter != alive_nodes.end(); iter++, index++) {
    for (size_t base = 0;; base++) {
      if (index + base * alive_nodes_size >= bundles.size()) {
        break;
      } else {
        schedule_map[bundles[index + base * alive_nodes_size]->BundleId()] = iter->first;
      }
    }
  }
  return schedule_map;
}

void GcsPlacementGroupScheduler::Schedule(
    std::shared_ptr<GcsPlacementGroup> placement_group,
    std::function<void(std::shared_ptr<GcsPlacementGroup>)> failure_callback,
    std::function<void(std::shared_ptr<GcsPlacementGroup>)> success_callback) {
  RAY_LOG(INFO) << "Scheduling placement group " << placement_group->GetName();
  auto bundles = placement_group->GetBundles();
  auto strategy = placement_group->GetStrategy();
  auto selected_nodes =
      scheduler_strategies_[strategy]->Schedule(bundles, GetScheduleContext());

  // If no nodes are available, scheduling fails.
  if (selected_nodes.empty()) {
    RAY_LOG(WARNING) << "Failed to schedule placement group "
                     << placement_group->GetName() << ", because no nodes are available.";
    failure_callback(placement_group);
    return;
  }

  // If schedule success, the decision will be set as schedule_map[bundles[pos]]
  // else will be set ClientID::Nil().
  auto bundle_locations = std::make_shared<BundleLocation>();
  // To count how many scheduler have been return, which include success and failure.
  auto finished_count = std::make_shared<size_t>();
  RAY_CHECK(
      placement_group_leasing_in_progress_.emplace(placement_group->GetPlacementGroupID())
          .second);

  /// TODO(AlisaWu): Change the strategy when reserve resource failed.
  for (auto &bundle : bundles) {
    const auto &bundle_id = bundle->BundleId();
    const auto &node_id = selected_nodes[bundle_id];
    RAY_CHECK(node_to_bundles_when_leasing_[node_id].emplace(bundle_id).second);

    ReserveResourceFromNode(
        bundle, gcs_node_manager_.GetNode(node_id),
        [this, bundle_id, bundle, bundles, node_id, placement_group, bundle_locations,
         finished_count, failure_callback, success_callback](const Status &status) {
          auto leasing_bundles = node_to_bundles_when_leasing_.find(node_id);
          RAY_CHECK(leasing_bundles != node_to_bundles_when_leasing_.end());
          auto bundle_iter = leasing_bundles->second.find(bundle->BundleId());
          RAY_CHECK(bundle_iter != leasing_bundles->second.end());
          // Remove the bundle from the leasing map as the reply is returned from the
          // remote node.
          leasing_bundles->second.erase(bundle_iter);
          if (leasing_bundles->second.empty()) {
            node_to_bundles_when_leasing_.erase(leasing_bundles);
          }

          if (status.ok()) {
            (*bundle_locations)[bundle_id] = std::make_pair(node_id, bundle);
          }

          if (++(*finished_count) == bundles.size()) {
            OnPlacementGroupResourceReserveDone(placement_group, bundles,
                                                bundle_locations, failure_callback,
                                                success_callback);
          }
        });
  }
}

void GcsPlacementGroupScheduler::DestroyPlacementGroupResources(
    const PlacementGroupID placement_group_id) {
  auto it = placement_group_to_bundle_location_.find(placement_group_id);
  RAY_CHECK(it != placement_group_to_bundle_location_.end());
  std::shared_ptr<BundleLocation> bundle_location = it->second;
  for (const auto &iter : *bundle_location) {
    auto &bundle_spec = iter.second.second;
    auto &node_id = iter.second.first;
    CancelResourceReserve(bundle_spec, gcs_node_manager_.GetNode(node_id));
  }
  placement_group_to_bundle_location_.erase(it);
}

void GcsPlacementGroupScheduler::CancelScheduling(
    const PlacementGroupID placement_group_id) {
  auto it = placement_group_leasing_in_progress_.find(placement_group_id);
  RAY_CHECK(it != placement_group_leasing_in_progress_.end());
  placement_group_leasing_in_progress_.erase(it);
}

void GcsPlacementGroupScheduler::ReserveResourceFromNode(
    const std::shared_ptr<BundleSpecification> &bundle,
    const std::shared_ptr<ray::rpc::GcsNodeInfo> &node, const StatusCallback &callback) {
  rpc::Address remote_address;
  remote_address.set_raylet_id(node->node_id());
  remote_address.set_ip_address(node->node_manager_address());
  remote_address.set_port(node->node_manager_port());
  auto node_id = ClientID::FromBinary(node->node_id());
  auto lease_client = GetOrConnectLeaseClient(remote_address);
  RAY_LOG(INFO) << "Leasing resource from node " << node_id
                << " for bundle: " << bundle->DebugString();
  lease_client->RequestResourceReserve(
      *bundle, [node_id, bundle, callback](
                   const Status &status, const rpc::RequestResourceReserveReply &reply) {
        // TODO(AlisaWu): Add placement group cancel.
        auto result = reply.success() ? Status::OK()
                                      : Status::IOError("Failed to reserve resource");
        if (result.ok()) {
          RAY_LOG(INFO) << "Finished leasing resource from " << node_id
                        << " for bundle: " << bundle->DebugString();
        } else {
          RAY_LOG(WARNING) << "Failed to lease resource from " << node_id
                           << " for bundle: " << bundle->DebugString();
        }
        callback(result);
      });
}

void GcsPlacementGroupScheduler::CancelResourceReserve(
    const std::shared_ptr<BundleSpecification> &bundle_spec,
    const std::shared_ptr<ray::rpc::GcsNodeInfo> &node) {
  if (node == nullptr) {
    return;
  }
  auto node_id = ClientID::FromBinary(node->node_id());
  RAY_LOG(INFO) << "Cancelling the resource reserved for bundle: "
                << bundle_spec->DebugString() << " at node " << node_id;
  rpc::Address remote_address;
  remote_address.set_raylet_id(node->node_id());
  remote_address.set_ip_address(node->node_manager_address());
  remote_address.set_port(node->node_manager_port());
  auto return_client = GetOrConnectLeaseClient(remote_address);
  return_client->CancelResourceReserve(
      *bundle_spec, [bundle_spec, node_id](const Status &status,
                                           const rpc::CancelResourceReserveReply &reply) {
        RAY_LOG(INFO) << "Finished cancelling the resource reserved for bundle: "
                      << bundle_spec->DebugString() << " at node " << node_id;
      });
}

std::shared_ptr<ResourceReserveInterface>
GcsPlacementGroupScheduler::GetOrConnectLeaseClient(const rpc::Address &raylet_address) {
  auto node_id = ClientID::FromBinary(raylet_address.raylet_id());
  auto iter = remote_lease_clients_.find(node_id);
  if (iter == remote_lease_clients_.end()) {
    auto lease_client = lease_client_factory_(raylet_address);
    iter = remote_lease_clients_.emplace(node_id, std::move(lease_client)).first;
  }
  return iter->second;
}

void GcsPlacementGroupScheduler::OnPlacementGroupResourceReserveDone(
    const std::shared_ptr<GcsPlacementGroup> &placement_group,
    const std::vector<std::shared_ptr<BundleSpecification>> &bundles,
    const std::shared_ptr<BundleLocation> &bundle_locations,
    const std::function<void(std::shared_ptr<GcsPlacementGroup>)>
        &schedule_failure_handler,
    const std::function<void(std::shared_ptr<GcsPlacementGroup>)>
        &schedule_success_handler) {
  const auto &placement_group_id = placement_group->GetPlacementGroupID();
  if (placement_group_leasing_in_progress_.find(placement_group_id) ==
          placement_group_leasing_in_progress_.end() ||
      bundle_locations->size() != bundles.size()) {
    // If the lease request has been already cancelled
    // or not every lease request succeeds,
    for (auto &iter : *bundle_locations) {
      auto &bundle_spec = iter.second.second;
      auto &node_id = iter.second.first;
      CancelResourceReserve(bundle_spec, gcs_node_manager_.GetNode(node_id));
    }
    schedule_failure_handler(placement_group);
  } else {
    // If we successfully created placement group, store them to GCS.
    rpc::ScheduleData data;
    for (const auto &iter : bundles) {
      // TODO(ekl) this is a hack to get a string key for the proto
      auto key = iter->PlacementGroupId().Hex() + "_" + std::to_string(iter->Index());
      data.mutable_schedule_plan()->insert(
          {key, (*bundle_locations)[iter->BundleId()].first.Binary()});
    }
    RAY_CHECK_OK(gcs_table_storage_->PlacementGroupScheduleTable().Put(
        placement_group_id, data,
        [schedule_success_handler, placement_group](Status status) {
          schedule_success_handler(placement_group);
        }));
    // Update `node_to_leased_bundles_`.
    for (const auto &iter : *bundle_locations) {
      const auto &location = iter.second;
      node_to_leased_bundles_[location.first].push_back(location.second);
    }
    RAY_CHECK(
        placement_group_to_bundle_location_.emplace(placement_group_id, bundle_locations)
            .second);
  }
  // Erase leasing in progress placement group.
  // This could've been removed if the leasing request is cancelled already.
  auto it = placement_group_leasing_in_progress_.find(placement_group_id);
  if (it != placement_group_leasing_in_progress_.end()) {
    placement_group_leasing_in_progress_.erase(it);
  }
}

std::unique_ptr<ScheduleContext> GcsPlacementGroupScheduler::GetScheduleContext() {
  // TODO(ffbin): We will add listener to the GCS node manager to handle node deletion.
  auto &alive_nodes = gcs_node_manager_.GetAllAliveNodes();
  for (const auto &iter : alive_nodes) {
    if (!node_to_leased_bundles_.contains(iter.first)) {
      node_to_leased_bundles_.emplace(
          iter.first, std::vector<std::shared_ptr<BundleSpecification>>());
    }
  }

  auto node_to_bundles = std::make_shared<absl::flat_hash_map<ClientID, int64_t>>();
  for (const auto &iter : node_to_leased_bundles_) {
    node_to_bundles->emplace(iter.first, iter.second.size());
  }
  return std::unique_ptr<ScheduleContext>(
      new ScheduleContext(node_to_bundles, gcs_node_manager_));
}

}  // namespace gcs
}  // namespace ray
