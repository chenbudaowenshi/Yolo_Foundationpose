from types import SimpleNamespace

from my_sim import head_control_mujoco


class _FakeNode:
    pass


_RosHeadControlNode = head_control_mujoco._make_node_class(
    _FakeNode,
    None,
    None,
    None,
    None,
)


def _endpoint(owner):
    return SimpleNamespace(node_name=owner[0], node_namespace=owner[1])


def _make_graph_node(*, standing, visible, publishers, subscriptions):
    node = object.__new__(_RosHeadControlNode)
    node.full_body_standing_mode = standing
    node.feedback_topic = "hardware/actuator_states"
    node.primary_command_topic = "hardware/actuators_cmds"
    node.command_topic = "hardware/actuators_cmds_override"
    node.state_machine_info_topic = "hardware/state_machine_info"
    node.motion_command_topic = "motion_commands"
    node.get_node_names_and_namespaces = lambda: list(visible)
    node.get_publishers_info_by_topic = lambda topic: list(
        publishers.get(topic, ())
    )
    node.get_subscriptions_info_by_topic = lambda topic: list(
        subscriptions.get(topic, ())
    )
    return node


def _valid_head_only_graph():
    hardware = head_control_mujoco.HEAD_ONLY_HARDWARE_NODE
    controller = head_control_mujoco.HEAD_ONLY_CONTROLLER_NODE
    return _make_graph_node(
        standing=False,
        visible=head_control_mujoco.HEAD_ONLY_HARDWARE_NODES,
        publishers={
            "hardware/actuator_states": [_endpoint(hardware)],
            "hardware/actuators_cmds": [_endpoint(controller)],
        },
        subscriptions={
            "hardware/actuators_cmds": [_endpoint(hardware)],
            "hardware/actuators_cmds_override": [_endpoint(controller)],
        },
    )


def _valid_standing_graph():
    hardware = head_control_mujoco.STANDING_HARDWARE_NODE
    controller = head_control_mujoco.STANDING_CONTROLLER_NODE
    remote = head_control_mujoco.REMOTE_CONTROLLER_NODE
    return _make_graph_node(
        standing=True,
        visible=head_control_mujoco.FULL_BODY_STANDING_NODES,
        publishers={
            "hardware/actuator_states": [_endpoint(hardware)],
            "hardware/actuators_cmds": [_endpoint(controller)],
            "hardware/state_machine_info": [_endpoint(controller)],
            "motion_commands": [_endpoint(remote)],
        },
        subscriptions={
            "hardware/actuators_cmds": [_endpoint(hardware)],
            "hardware/actuators_cmds_override": [_endpoint(controller)],
            "motion_commands": [_endpoint(controller)],
        },
    )


def test_valid_head_only_graph_is_accepted():
    ready, detail = _valid_head_only_graph()._hardware_graph_status()
    assert ready is True
    assert detail == "isolated head-only ROS graph verified"


def test_valid_head_enabled_standing_graph_is_accepted():
    ready, detail = _valid_standing_graph()._hardware_graph_status()
    assert ready is True
    assert detail == "head-enabled full-body ROS graph verified"


def test_current_head_disabled_robot_graph_is_rejected():
    node = _make_graph_node(
        standing=True,
        visible={
            ("hardware_elf3", "/"),
            head_control_mujoco.STANDING_CONTROLLER_NODE,
            head_control_mujoco.REMOTE_CONTROLLER_NODE,
        },
        publishers={},
        subscriptions={},
    )
    ready, detail = node._hardware_graph_status()
    assert ready is False
    assert "hardware_elf3_head" in detail


def test_wrong_override_subscriber_is_rejected():
    node = _valid_standing_graph()
    original = node.get_subscriptions_info_by_topic
    node.get_subscriptions_info_by_topic = lambda topic: (
        [_endpoint(("unexpected_controller", "/"))]
        if topic == "hardware/actuators_cmds_override"
        else original(topic)
    )
    ready, detail = node._hardware_graph_status()
    assert ready is False
    assert "override subscriber" in detail


def test_duplicate_endpoint_from_same_owner_is_rejected():
    node = _valid_standing_graph()
    original = node.get_publishers_info_by_topic
    node.get_publishers_info_by_topic = lambda topic: (
        [
            _endpoint(head_control_mujoco.STANDING_HARDWARE_NODE),
            _endpoint(head_control_mujoco.STANDING_HARDWARE_NODE),
        ]
        if topic == "hardware/actuator_states"
        else original(topic)
    )
    ready, detail = node._hardware_graph_status()
    assert ready is False
    assert "hardware feedback publisher" in detail


def test_standing_graph_requires_remote_motion_link():
    node = _valid_standing_graph()
    original = node.get_publishers_info_by_topic
    node.get_publishers_info_by_topic = lambda topic: (
        [] if topic == "motion_commands" else original(topic)
    )
    ready, detail = node._hardware_graph_status()
    assert ready is False
    assert "remote motion publisher" in detail
