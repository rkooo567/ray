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

#include "ray/gcs/gcs_server/gcs_placement_group_manager.h"

#include "ray/common/ray_config.h"
#include "ray/gcs/pb_util.h"
#include "ray/util/asio_util.h"
#include "src/ray/protobuf/gcs.pb.h"

namespace ray {
namespace gcs {

void GcsPlacementGroup::UpdateState(
    rpc::PlacementGroupTableData::PlacementGroupState state) {
  placement_group_table_data_.set_state(state);
}

rpc::PlacementGroupTableData::PlacementGroupState GcsPlacementGroup::GetState() const {
  return placement_group_table_data_.state();
}

PlacementGroupID GcsPlacementGroup::GetPlacementGroupID() const {
  return PlacementGroupID::FromBinary(placement_group_table_data_.placement_group_id());
}

std::string GcsPlacementGroup::GetName() const {
  return placement_group_table_data_.name();
}

std::vector<std::shared_ptr<BundleSpecification>> GcsPlacementGroup::GetBundles() const {
  auto bundles = placement_group_table_data_.bundles();
  std::vector<std::shared_ptr<BundleSpecification>> ret_bundles;
  for (auto &bundle : bundles) {
    ret_bundles.push_back(std::make_shared<BundleSpecification>(bundle));
  }
  return ret_bundles;
}

rpc::PlacementStrategy GcsPlacementGroup::GetStrategy() const {
  return placement_group_table_data_.strategy();
}

const rpc::PlacementGroupTableData &GcsPlacementGroup::GetPlacementGroupTableData() {
  return placement_group_table_data_;
}

const std::string GcsPlacementGroup::DebugString() const {
  std::stringstream stream;
  stream << "placement group id = " << GetPlacementGroupID() << ", name = " << GetName()
         << ", strategy = " << GetStrategy();
  return stream.str();
}

/////////////////////////////////////////////////////////////////////////////////////////

GcsPlacementGroupManager::GcsPlacementGroupManager(
    boost::asio::io_context &io_context,
    std::shared_ptr<GcsPlacementGroupSchedulerInterface> scheduler,
    std::shared_ptr<gcs::GcsTableStorage> gcs_table_storage)
    : io_context_(io_context),
      gcs_placement_group_scheduler_(std::move(scheduler)),
      gcs_table_storage_(std::move(gcs_table_storage)) {}

void GcsPlacementGroupManager::RegisterPlacementGroup(
    const ray::rpc::CreatePlacementGroupRequest &request, StatusCallback callback) {
  RAY_CHECK(callback);
  const auto &placement_group_spec = request.placement_group_spec();
  auto placement_group_id =
      PlacementGroupID::FromBinary(placement_group_spec.placement_group_id());
  auto placement_group = std::make_shared<GcsPlacementGroup>(request);

  // TODO(ffbin): If GCS is restarted, GCS client will repeatedly send
  // `CreatePlacementGroup` requests,
  //  which will lead to resource leakage, we will solve it in next pr.
  // Mark the callback as pending and invoke it after the placement_group has been
  // successfully created.
  placement_group_to_register_callback_[placement_group_id] = std::move(callback);
  pending_placement_groups_.emplace_back(std::move(placement_group));
  SchedulePendingPlacementGroups();
}

PlacementGroupID GcsPlacementGroupManager::GetPlacementGroupIDByName(
    const std::string &name) {
  PlacementGroupID placement_group_id = PlacementGroupID::Nil();
  for (const auto &iter : registered_placement_groups_) {
    if (iter.second->GetName() == name) {
      placement_group_id = iter.first;
      break;
    }
  }
  return placement_group_id;
}

void GcsPlacementGroupManager::OnPlacementGroupCreationFailed(
    std::shared_ptr<GcsPlacementGroup> placement_group) {
  RAY_LOG(WARNING) << "Failed to create placement group " << placement_group->GetName()
                   << ", try again.";
  // We will attempt to schedule this placement_group once an eligible node is
  // registered.
  pending_placement_groups_.emplace_back(std::move(placement_group));
  scheduling_progress_.id = PlacementGroupID::Nil();
  scheduling_progress_.is_creating = false;
  RetryCreatingPlacementGroup();
}

void GcsPlacementGroupManager::OnPlacementGroupCreationSuccess(
    const std::shared_ptr<GcsPlacementGroup> &placement_group) {
  RAY_LOG(INFO) << "Successfully created placement group " << placement_group->GetName();
  placement_group->UpdateState(rpc::PlacementGroupTableData::CREATED);
  auto placement_group_id = placement_group->GetPlacementGroupID();
  RAY_CHECK_OK(gcs_table_storage_->PlacementGroupTable().Put(
      placement_group_id, placement_group->GetPlacementGroupTableData(),
      [this, placement_group_id](Status status) {
        RAY_CHECK_OK(status);

        // Invoke callback for registration request of this placement_group
        // and remove it from placement_group_to_register_callback_.
        auto iter = placement_group_to_register_callback_.find(placement_group_id);
        if (iter != placement_group_to_register_callback_.end()) {
          iter->second(Status::OK());
          placement_group_to_register_callback_.erase(iter);
        }
        scheduling_progress_.id = PlacementGroupID::Nil();
        scheduling_progress_.is_creating = false;
        auto &placement_group_it = registered_placement_groups_.find(placement_group_id);
        // Add an entry to the created map if placement group is not removed at this
        // point.
        if (placement_group_it != registered_placement_groups_.end()) {
          created_placement_group_.emplace(placement_group_id,
                                           placement_group_it->second);
        }
        SchedulePendingPlacementGroups();
      }));
}

void GcsPlacementGroupManager::SchedulePendingPlacementGroups() {
  if (pending_placement_groups_.empty() || scheduling_progress_.is_creating) {
    return;
  }
  const auto &placement_group = pending_placement_groups_.pop_front();
  scheduling_progress_.id = placement_group->GetPlacementGroupID();
  scheduling_progress_.is_creating = true;
  gcs_placement_group_scheduler_->Schedule(
      placement_group,
      [this](std::shared_ptr<GcsPlacementGroup> placement_group) {
        OnPlacementGroupCreationFailed(std::move(placement_group));
      },
      [this](std::shared_ptr<GcsPlacementGroup> placement_group) {
        OnPlacementGroupCreationSuccess(std::move(placement_group));
      });
}
}  // namespace gcs

void GcsPlacementGroupManager::HandleCreatePlacementGroup(
    const ray::rpc::CreatePlacementGroupRequest &request,
    ray::rpc::CreatePlacementGroupReply *reply,
    ray::rpc::SendReplyCallback send_reply_callback) {
  auto placement_group_id =
      PlacementGroupID::FromBinary(request.placement_group_spec().placement_group_id());
  const auto &name = request.placement_group_spec().name();
  const auto &strategy = request.placement_group_spec().strategy();
  auto placement_group = std::make_shared<GcsPlacementGroup>(request);
  RAY_LOG(INFO) << "Registering placement group, " << placement_group.DebugString();
  // NOTE: Registration should be called here. Otherwise, there's no way to
  registered_placement_groups_.emplace(placement_group_id, placement_group);
  RAY_CHECK_OK(gcs_table_storage_->PlacementGroupTable().Put(
      placement_group_id, placement_group->GetPlacementGroupTableData(),
      [this, request, reply, send_reply_callback, placement_group_id, name,
       strategy](Status status) {
        RAY_CHECK_OK(status);
        if (registered_placement_groups_.find(placement_group_id) ==
            registered_placement_groups_.end()) {
          std::stringstream stream;
          stream << "Placement group of id " << placement_group_id
                 << " has been removed.";
          GCS_RPC_SEND_REPLY(send_reply_callback, reply, Status::OK(stream.str()));
          return;
        }

        RegisterPlacementGroup(request, [reply, send_reply_callback, placement_group_id,
                                         name, strategy](Status status) {
          if (status.ok()) {
            RAY_LOG(INFO) << "Finished registering placement group, "
                          << placement_group.DebugString();
          } else {
            RAY_LOG(INFO) << "Failed to register placement group, "
                          << placement_group.DebugString();
          }
          GCS_RPC_SEND_REPLY(send_reply_callback, reply, status);
        });
      }));
}

void GcsPlacementGroupManager::HandleRemovePlacementGroup(
    const rpc::RemovePlacementGroupRequest &request,
    rpc::RemovePlacementGroupReply *reply, rpc::SendReplyCallback send_reply_callback) {
  const auto placement_group_id =
      PlacementGroupID::FromBinary(request.placement_group_id());

  RemovePlacementGroup(placement_group_id, [send_reply_callback, reply,
                                            placement_group_id](Status status) {
    if (status.ok()) {
      RAY_LOG(ERROR) << "Placement group of an id, " << placement_group_id
                     << " is removed successfully.";
    } else {
      RAY_LOG(ERROR) << "Removing a placement group of an id, " << placement_group_id
                     << " failed.";
    }
    GCS_RPC_SEND_REPLY(send_reply_callback, reply, status);
  });
}

void GcsPlacementGroupManager::RemovePlacementGroup(
    const PlacementGroupID &placement_group_id,
    StatusCallback on_placement_group_removed) {
  RAY_CHECK(on_placement_group_removed);
  if (registered_placement_groups_.find(placement_group_id) ==
      registered_placement_groups_.end()) {
    std::stringstream stream;
    stream << "Placement group of id " << placement_group_id << " is already removed";
    on_placement_group_removed(Status::Invalid(stream.str()));
  }

  registered_placement_groups_.erase(placement_group_id);
  auto &it = placement_group_to_register_callback_.find(placement_group_id);
  if (it != placement_group_to_register_callback_.end()) {
    std::stringstream stream;
    stream << "Placement group of id " << placement_group_id << " is removed";
    it->second(Status::NotFound(stream.str()));
    placement_group_to_register_callback_.erase(it);
  }

  if (is_creating_.first == placement_group_id) {
    // If placement group is scheduling.
    placement_group_scheduling_in_progress.gcs_placement_group_scheduler_
        ->CancelSchedulingIfNeeded(placement_group_id);
  } else if (created_placement_group_.find(placement_group_id) !=
             created_placement_group_.end()) {
    // If placement gropu is already created.
    DestroyPlacementGroupResources();
  } else {
    // If placement group is pending
    auto pending_it = std::find_if(
        pending_placement_groups_.begin(), pending_placement_groups_.end(),
        [placement_group_id](const std::shared_ptr<GcsPlacementGroup> &placement_group) {
          return placement_group->GetPlacementGroupID() == placement_group_id;
        });

    // The placement group was pending scheduling, remove it from the queue.
    RAY_CHECK(pending_it != pending_placement_groups_.end());
    pending_placement_groups_.erase(pending_it);
  }

  on_placement_group_removed(Status::OK());
}

void GcsPlacementGroupManager::RetryCreatingPlacementGroup() {
  execute_after(io_context_, [this] { SchedulePendingPlacementGroups(); },
                RayConfig::instance().gcs_create_placement_group_retry_interval_ms());
}

}  // namespace ray
}  // namespace ray
