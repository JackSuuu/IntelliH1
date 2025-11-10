#pragma once
#include <vector>

struct Point {
    double x, y;
};

struct NavigationCommand {
    double linear_velocity;   // m/s
    double angular_velocity;  // rad/s
};

class PointCloudProcessor {
public:
    PointCloudProcessor(double max_range);
    std::vector<Point> lidar_to_point_cloud(const std::vector<double>& lidar_readings, double fov_rad);
    std::vector<Point> remove_noise(const std::vector<Point>& cloud, double min_dist_sq);
    
    // Filter obstacles that are in the path towards goal
    std::vector<Point> filter_obstacles_in_path(
        const std::vector<Point>& cloud, 
        double heading_to_goal,
        double fov_angle = M_PI / 3.0  // 60 degree FOV
    );
    
    Point get_avoidance_vector(const std::vector<Point>& cloud);
    
    // Main navigation computation: combines goal-seeking + obstacle avoidance
    NavigationCommand compute_navigation_command(
        const Point& current_pos,
        double current_heading,
        const Point& goal_pos,
        const std::vector<Point>& obstacles
    );

private:
    double max_range_;
};
