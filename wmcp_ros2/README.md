# WMCP ROS 2 Integration

WMCP protocol nodes for ROS 2 robot communication.

## Architecture

```
┌──────────────┐    /wmcp/messages    ┌──────────────┐
│ V-JEPA Agent │ ──── WMCPMessage ───→│  Subscriber  │
│  (Publisher)  │                      │   (Decoder)  │
└──────────────┘                      └──────────────┘
┌──────────────┐    /wmcp/messages
│ DINOv2 Agent │ ──── WMCPMessage ───→   (same topic)
│  (Publisher)  │
└──────────────┘
```

## Quick Start

```bash
# Without ROS 2 (pure Python simulation)
python -m wmcp_ros2.demo

# With ROS 2 (requires rclpy)
ros2 run wmcp_ros2 publisher_node
ros2 run wmcp_ros2 subscriber_node
```

## Message Format

```
WMCPMessage:
  agent_id: 0
  version: "0.1.0"
  K: 3              # vocab size
  L: 2              # positions per agent
  tokens: [1, 2]    # discrete symbols
  encoder: "vjepa2"
  domain: "physics_spring"
```

Wire format: JSON over std_msgs/String (ROS 2) or queue (simulation).

## Files

| File | Description |
|------|-------------|
| `wmcp_msg.py` | Message dataclass + ROS 2 .msg definition |
| `publisher_node.py` | Agent encoder + publisher |
| `subscriber_node.py` | Message collector + decoder |
| `demo.py` | End-to-end communication demo |

## Installation (with ROS 2)

```bash
# In your ROS 2 workspace
cd ~/ros2_ws/src
ln -s /path/to/wmcp_ros2 .
cd ~/ros2_ws
colcon build --packages-select wmcp_ros2
source install/setup.bash
```
