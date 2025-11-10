#include "point_cloud_processor.h"
#include <cmath>
#include <numeric>

PointCloudProcessor::PointCloudProcessor(double max_range) : max_range_(max_range) {}

std::vector<Point> PointCloudProcessor::lidar_to_point_cloud(const std::vector<double>& lidar_readings, double fov_rad) {
    std::vector<Point> cloud;
    double angle_increment = fov_rad / (lidar_readings.size() - 1);
    double start_angle = -fov_rad / 2.0;

    for (size_t i = 0; i < lidar_readings.size(); ++i) {
        double distance = lidar_readings[i];
        if (distance < max_range_) {
            double angle = start_angle + i * angle_increment;
            cloud.push_back({distance * std::cos(angle), distance * std::sin(angle)});
        }
    }
    return cloud;
}

std::vector<Point> PointCloudProcessor::remove_noise(const std::vector<Point>& cloud, double min_dist_sq) {
    if (cloud.size() < 2) {
        return cloud;
    }
    
    std::vector<Point> filtered_cloud;
    // A simple filter: remove points that are too far from their neighbors
    // This is a naive implementation. A real implementation would use something like a VoxelGrid or StatisticalOutlierRemoval filter.
    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& p1 = cloud[i];
        bool has_neighbor = false;
        for (size_t j = 0; j < cloud.size(); ++j) {
            if (i == j) continue;
            const auto& p2 = cloud[j];
            double dist_sq = std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2);
            if (dist_sq < min_dist_sq) {
                has_neighbor = true;
                break;
            }
        }
        if (has_neighbor) {
            filtered_cloud.push_back(p1);
        }
    }
    return filtered_cloud;
}

// Filter obstacles: only keep those that are in the direction of movement
std::vector<Point> PointCloudProcessor::filter_obstacles_in_path(
    const std::vector<Point>& cloud, 
    double heading_to_goal,
    double fov_angle) {
    
    std::vector<Point> filtered;
    for (const auto& p : cloud) {
        double angle_to_obstacle = std::atan2(p.y, p.x);
        double angle_diff = std::abs(angle_to_obstacle - heading_to_goal);
        
        // Normalize to [-pi, pi]
        while (angle_diff > M_PI) angle_diff -= 2.0 * M_PI;
        while (angle_diff < -M_PI) angle_diff += 2.0 * M_PI;
        
        // Only consider obstacles within FOV of goal direction
        if (std::abs(angle_diff) < fov_angle / 2.0) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

Point PointCloudProcessor::get_avoidance_vector(const std::vector<Point>& cloud) {
    Point avoidance_vector = {0.0, 0.0};
    if (cloud.empty()) {
        return avoidance_vector;
    }

    for (const auto& p : cloud) {
        double dist_sq = p.x * p.x + p.y * p.y;
        if (dist_sq > 1e-6) {
            double repulsive_force = 1.0 / dist_sq;
            avoidance_vector.x -= repulsive_force * p.x;
            avoidance_vector.y -= repulsive_force * p.y;
        }
    }

    double norm = std::sqrt(avoidance_vector.x * avoidance_vector.x + avoidance_vector.y * avoidance_vector.y);
    if (norm > 1.0) {
        avoidance_vector.x /= norm;
        avoidance_vector.y /= norm;
    }

    return avoidance_vector;
}

NavigationCommand PointCloudProcessor::compute_navigation_command(
    const Point& current_pos,
    double current_heading,
    const Point& goal_pos,
    const std::vector<Point>& obstacles
) {
    NavigationCommand cmd = {0.0, 0.0};
    
    // Anti-stuck mechanism: track if we're stuck
    static Point last_pos = {0.0, 0.0};
    static int stuck_count = 0;
    static int escape_mode_counter = 0;
    
    double moved = std::sqrt(
        std::pow(current_pos.x - last_pos.x, 2) + 
        std::pow(current_pos.y - last_pos.y, 2)
    );
    
    if (moved < 0.05) {  // Moved less than 5cm
        stuck_count++;
    } else {
        stuck_count = 0;
        escape_mode_counter = 0;  // Reset escape mode
    }
    
    last_pos = current_pos;
    
    // If stuck for too long, enter escape mode
    if (stuck_count > 30 && escape_mode_counter < 100) {  // Stuck for 3 seconds
        escape_mode_counter++;
        // Try to escape: back up and turn
        cmd.linear_velocity = -0.10;  // Back up
        cmd.angular_velocity = (escape_mode_counter % 2 == 0) ? 0.4 : -0.4;  // Oscillate
        printf("    [ESCAPE MODE] Backing up to escape (stuck_count=%d)\n", stuck_count);
        return cmd;
    }
    
    // 1. Calculate distance and direction to goal
    double to_goal_x = goal_pos.x - current_pos.x;
    double to_goal_y = goal_pos.y - current_pos.y;
    double distance = std::sqrt(to_goal_x * to_goal_x + to_goal_y * to_goal_y);
    
    // Debug output every 100 calls
    static int call_count = 0;
    if (++call_count % 100 == 0) {
        printf("[C++ DEBUG] Pos:(%.2f,%.2f) Goal:(%.2f,%.2f) Dist:%.2fm Obstacles:%zu\n", 
               current_pos.x, current_pos.y, goal_pos.x, goal_pos.y, distance, obstacles.size());
    }
    
    // Stop if at waypoint/goal (small threshold for waypoint following)
    if (distance < 0.3) {  // Changed from 1.5 to 0.3 for accurate waypoint tracking
        if (call_count % 50 == 0) {
            printf("[C++ SUCCESS] Reached waypoint! Distance: %.2fm\n", distance);
        }
        return cmd;
    }
    
    // 2. Desired heading to goal  
    double desired_heading = std::atan2(to_goal_y, to_goal_x);
    
    // Heading error (normalized to [-pi, pi])
    double heading_error = desired_heading - current_heading;
    while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
    while (heading_error < -M_PI) heading_error += 2.0 * M_PI;
    
    double abs_heading_error = std::abs(heading_error);
    
    // 3. Smart obstacle detection - only consider obstacles in our path
    // Obstacles are in car's local coordinate system (front = +x)
    // We want obstacles within ±60° of straight ahead (0 radians in car frame)
    std::vector<Point> relevant_obstacles = filter_obstacles_in_path(obstacles, 0.0, M_PI * 2.0 / 3.0);  // ±60°
    
    // Find closest obstacle in path
    double min_obstacle_dist = 1000.0;
    for (const auto& obs : relevant_obstacles) {
        double dist = std::sqrt(obs.x * obs.x + obs.y * obs.y);
        if (dist < min_obstacle_dist) {
            min_obstacle_dist = dist;
        }
    }
    
    bool has_obstacle = (min_obstacle_dist < 2.0);  // Obstacle within 2m
    
    // Debug output
    if (call_count % 100 == 0) {
        printf("    Heading: Desired:%.0f° Current:%.0f° Error:%.0f° | Obstacle: %s (%.2fm)\n",
               desired_heading * 180.0 / M_PI,
               current_heading * 180.0 / M_PI,
               heading_error * 180.0 / M_PI,
               has_obstacle ? "YES" : "NO",
               min_obstacle_dist);
    }
    
    // 4. Advanced Pure Pursuit Control
    
    // === STRATEGY ===
    // 1. If obstacle very close (<1.0m): STOP and turn away
    // 2. If obstacle close (<2.0m): SLOW and gentle turn
    // 3. If far from goal (>3.0m): Normal speed, moderate turns
    // 4. If approaching goal (<3.0m): SLOW down progressively
    // 5. If very close to goal (<2.0m): EXTRA slow, minimal turns
    
    if (has_obstacle && min_obstacle_dist < 0.8) {
        // EMERGENCY: Very close obstacle - slow down but maintain momentum
        cmd.linear_velocity = 0.10;  // Increased from 0.05 to maintain momentum
        
        // Turn more aggressively to avoid obstacle
        // If obstacle is dead ahead, turn perpendicular to goal direction
        double avoid_angular;
        if (abs_heading_error < M_PI / 4) {
            // Heading roughly towards goal, turn away from obstacle
            avoid_angular = (heading_error > 0) ? 0.45 : -0.45;  // Increased turn rate
        } else {
            // Already turned, try to go around
            avoid_angular = 0.35 * heading_error;  // Increased from 0.3
        }
        cmd.angular_velocity = avoid_angular;
        
        // Limit
        if (cmd.angular_velocity > 0.40) cmd.angular_velocity = 0.40;  // Increased from 0.35
        if (cmd.angular_velocity < -0.40) cmd.angular_velocity = -0.40;
        
        if (call_count % 100 == 0) {
            printf("    [EMERGENCY] Obstacle at %.2fm, creeping around...\n", min_obstacle_dist);
        }
    }
    else if (has_obstacle && min_obstacle_dist < 1.5) {
        // CAUTION: Obstacle nearby - slow down and navigate around
        cmd.linear_velocity = 0.15;  // Increased from 0.12 for better momentum
        
        // Moderate turn towards goal with obstacle awareness
        double angular_gain = 0.25;  // Increased from 0.20 for more responsive turning
        cmd.angular_velocity = angular_gain * heading_error;
        
        // Limit
        if (cmd.angular_velocity > 0.30) cmd.angular_velocity = 0.30;  // Increased from 0.25
        if (cmd.angular_velocity < -0.30) cmd.angular_velocity = -0.30;
        
        if (call_count % 100 == 0) {
            printf("    [CAUTION] Obstacle at %.2fm, navigating around...\n", min_obstacle_dist);
        }
    }
    else {
        // NORMAL NAVIGATION: Pure pursuit towards goal
        
        // Strategy: If heading error is large, SLOW DOWN and turn more
        // This prevents wide arcs and allows tighter turns
        
        if (abs_heading_error > M_PI / 2.0) {
            // Heading error > 90°: Need to turn around
            // SLOW DOWN to allow tight turn
            cmd.linear_velocity = 0.08;  // Very slow while turning
            
            // More aggressive turning
            double angular_gain = 0.40;
            cmd.angular_velocity = angular_gain * heading_error;
            
            // Higher angular velocity limit for U-turns
            double max_angular = 0.50;
            if (cmd.angular_velocity > max_angular) cmd.angular_velocity = max_angular;
            if (cmd.angular_velocity < -max_angular) cmd.angular_velocity = -max_angular;
            
            if (call_count % 100 == 0) {
                printf("    [NAVIGATE] Large heading error %.0f°, turning sharply\n", 
                       abs_heading_error * 180.0 / M_PI);
            }
        }
        else if (abs_heading_error > M_PI / 4.0) {
            // Heading error > 45°: Moderate correction needed
            cmd.linear_velocity = 0.15;  // Slow while turning
            
            double angular_gain = 0.30;
            cmd.angular_velocity = angular_gain * heading_error;
            
            double max_angular = 0.40;
            if (cmd.angular_velocity > max_angular) cmd.angular_velocity = max_angular;
            if (cmd.angular_velocity < -max_angular) cmd.angular_velocity = -max_angular;
        }
        else {
            // Heading error < 45°: Can go faster
            // Linear velocity strategy based on distance
            if (distance > 5.0) {
                cmd.linear_velocity = 0.30;  // Full speed when far
            } else if (distance > 3.0) {
                cmd.linear_velocity = 0.20;
            } else if (distance > 2.0) {
                cmd.linear_velocity = 0.15;
            } else {
                cmd.linear_velocity = 0.10;  // Slow near goal
            }
            
            // Angular velocity: gentler when well-aligned
            double angular_gain;
            double max_angular;
            
            if (distance > 3.0) {
                angular_gain = 0.25;
                max_angular = 0.30;
            } else if (distance > 2.0) {
                angular_gain = 0.18;
                max_angular = 0.22;
            } else {
                angular_gain = 0.10;
                max_angular = 0.12;
            }
            
            cmd.angular_velocity = angular_gain * heading_error;
            if (cmd.angular_velocity > max_angular) cmd.angular_velocity = max_angular;
            if (cmd.angular_velocity < -max_angular) cmd.angular_velocity = -max_angular;
            
            if (call_count % 100 == 0) {
                printf("    [NAVIGATE] v=%.2f ω=%.2f (aligned)\n", 
                       cmd.linear_velocity, cmd.angular_velocity);
            }
        }
    }
    
    return cmd;
}
