import asyncio
from rclpy.node import Node
from geometry_msgs.msg import Twist

message_queue = asyncio.Queue()

class MoveAgent(Node):
    """ROS2 Movement Agent for controlling the robot's movement."""

    def __init__(self):
        super().__init__('move_bot_agent')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("MoveBotAgent initialized.")

    async def move(self, command, duration=2.0):
        """Executes a movement command for a given duration."""
        cmd = Twist()
        if command == 'forward':
            cmd.linear.x = -0.6
        elif command == 'backward':
            cmd.linear.x = 0.6
        elif command == 'left':
            cmd.angular.z = 1.0
        elif command == 'right':
            cmd.angular.z = -1.0
        elif command == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            self.get_logger().warn(f'Unknown command: {command}')
            return

        self.get_logger().info(f'Executing {command} for {duration} seconds')

        for _ in range(int(duration / 0.2)):
            self.publisher.publish(cmd)
            await asyncio.sleep(0.2)

        stop_cmd = Twist()
        self.publisher.publish(stop_cmd)
        self.get_logger().info(f'{command} completed')

        await message_queue.put(f"MoveBotAgent: Executed {command}")