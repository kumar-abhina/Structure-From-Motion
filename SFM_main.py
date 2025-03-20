import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gtsam
import gtsam.utils.plot as gtsam_plot
import open3d as o3d

class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def create_pointcloud(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    @staticmethod
    def display_pointcloud(cloud: np.ndarray, cloud_colors: np.ndarray):
        mask = np.linalg.norm(cloud, axis=1) < 1000
        cloud = cloud[mask]
        cloud_colors = cloud_colors[mask] / 255.0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=5, c=cloud_colors, marker='o', label='Pointcloud')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.legend()
        plt.show()

    def display_scene(camera_poses: dict, cloud: np.ndarray = None, cloud_colors: np.ndarray = None, 
                  show_cameras=True, show_pointcloud=True, limit_far_points=True,
                  rectangle_size=2, frustum_depth=10, axis_length=5, title: str = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title, fontsize=16)  # Add the header label as the figure title

        # Process camera poses
        if show_cameras:
            positions = [pose[:3, 3] for pose in camera_poses.values()]
            orientations = [pose[:3, :3] for pose in camera_poses.values()]

            for i, (position, orientation) in enumerate(zip(positions, orientations)):
                ax.scatter(position[0], position[1], position[2], c='r', marker='o', label=f'Camera {i}' if i == 0 else "")
                ax.text(position[0], position[1], position[2], f'{i}', size=8, color='k')

                # Draw orientation axes
                x_axis = orientation.dot(np.array([axis_length, 0, 0]))
                y_axis = orientation.dot(np.array([0, axis_length, 0]))
                z_axis = orientation.dot(np.array([0, 0, axis_length]))
                ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r')
                ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g')
                ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b')

                # Draw a black frustum representing the camera
                near_plane = rectangle_size / 2
                far_plane = rectangle_size
                corners = np.array([
                    [-near_plane, -near_plane, 0],
                    [near_plane, -near_plane, 0],
                    [near_plane, near_plane, 0],
                    [-near_plane, near_plane, 0],
                    [-far_plane, -far_plane, frustum_depth],
                    [far_plane, -far_plane, frustum_depth],
                    [far_plane, far_plane, frustum_depth],
                    [-far_plane, far_plane, frustum_depth]
                ])
                rotated_corners = np.dot(orientation, corners.T).T + position

                # Define the faces of the frustum
                faces = [
                    [rotated_corners[j] for j in [0, 1, 2, 3]],  # Near plane
                    [rotated_corners[j] for j in [4, 5, 6, 7]],  # Far plane
                    [rotated_corners[j] for j in [0, 1, 5, 4]],  # Left
                    [rotated_corners[j] for j in [1, 2, 6, 5]],  # Bottom
                    [rotated_corners[j] for j in [2, 3, 7, 6]],  # Right
                    [rotated_corners[j] for j in [3, 0, 4, 7]]   # Top
                ]
                ax.add_collection3d(Poly3DCollection(faces, color='black', alpha=0.3))

        # Process point cloud
        if show_pointcloud and cloud is not None and cloud_colors is not None:
            if limit_far_points:
                mask = np.linalg.norm(cloud, axis=1) < 1000
                cloud = cloud[mask]
                cloud_colors = cloud_colors[mask]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=5, c=cloud_colors / 255.0, marker='o', label='Point Cloud')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.legend()
        plt.show()


class PointCloud_Capture:
    class Point3D:
        def __init__(self, coords, origin):
            self.coords = coords
            self.origin = origin

    def __init__(self, points_3D, index, keypoints1, keypoints2):
        self.cloud = [self.Point3D(points_3D[i], {index: keypoints1[i], index + 1: keypoints2[i]}) for i in range(len(points_3D))]

    def find_point(self, point, index):
        for existing_point in self.cloud:
            if index in existing_point.origin and np.array_equal(point.origin[index], existing_point.origin[index]):
                return True, existing_point
        return False, None

    def add_or_update(self, index, points_3D, keypoints1, keypoints2):
        for i, coords in enumerate(points_3D):
            new_point = self.Point3D(coords, {index: keypoints1[i], index + 1: keypoints2[i]})
            found, existing_point = self.find_point(new_point, index)
            if found:
                existing_point.origin[index + 1] = keypoints2[i]
            else:
                self.cloud.append(new_point)

    def match_points(self, index, previous_keypoints, current_keypoints):
        matched_2d_points = []
        matched_3d_points = []
        matched_indices = []

        for i, current_point in enumerate(current_keypoints):
            if any(np.array_equal(current_point, prev_point) for prev_point in previous_keypoints):
                matched_2d_points.append(current_point)
                matched_indices.append(i)

        for matched_point in matched_2d_points:
            for point_3d in self.cloud:
                if index in point_3d.origin and np.array_equal(matched_point, point_3d.origin[index]):
                    matched_3d_points.append(point_3d.coords)
                    break

        matched_2d_points = np.array(matched_2d_points, dtype=np.float32)
        matched_3d_points = np.array(matched_3d_points, dtype=np.float32)

        if matched_2d_points.shape[0] != matched_3d_points.shape[0]:
            print("Error: Mismatch in matched 2D and 3D points")
            print(f"2D Points: {matched_2d_points.shape}, 3D Points: {matched_3d_points.shape}")
            print(f"Previous Keypoints: {previous_keypoints.shape}, Current Keypoints: {current_keypoints.shape}")
            print(f"Index: {index}")
            exit()

        return matched_2d_points, matched_3d_points, matched_indices

    def extract_coordinates(self):
        return np.array([point.coords for point in self.cloud])



class SFM_LOOP(object): 
    def __init__(self, data_dir, descriptor_name="SIFT", num_features=5000, matcher="FLANN", show_matches=True, 
                 show_keypoints=False, use_clahe=False, use_BA=False, plot_matching=False, make_dense=False):
        
        super(SFM_LOOP, self).__init__()
        self.data_dir = data_dir
        self.num_features = num_features
        self.show_matches = show_matches
        self.show_keypoints = show_keypoints
        self.use_clahe = use_clahe
        self.dist_coeff = np.zeros((4, 1))
        self.use_BA = use_BA
        self.make_dense = make_dense
        self.plot_matching = plot_matching
        self.descriptor_name = descriptor_name
                
        # Variable to speed up computation
        self.keypoints1 = None
        self.descriptors1 = None
        
        # Store the pointcloud and camera poses
        self.pointcloud = None
        self.camera_poses = {}
        
        # Make variables for descriptor and matcher
        if descriptor_name == "ORB":
            self.descriptor = cv2.ORB_create(num_features)
        elif descriptor_name == "SIFT":
            self.descriptor = cv2.SIFT_create(num_features, nOctaveLayers=3, contrastThreshold=0.04)
        
        if matcher == "FLANN":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher == "BFMatcher":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    def read_images(self):
        Images = {}
        image_list = sorted(os.listdir(self.data_dir))
        N = len(image_list)

        for i, image_file in enumerate(image_list):
            image_path = os.path.join(self.data_dir, image_file)
            print(f"Reading image from Folder{i}: {image_file}")
            color_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

            Images[f'image{i}_color'] = color_image
            Images[f'image{i}'] = gray_image

        return Images, N
    
    def non_maximum_suppression(self, points):
        height, width = self.Images['image0'].shape[:2]
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        responses = np.array([point.response for point in points])
        mask = np.flip(np.argsort(responses))
        point_list = np.rint([point.pt for point in points])[mask].astype(int)
        nms_mask = []
        for point, index in zip(point_list, mask):
            if binary_mask[point[1], point[0]] == 0:
                nms_mask.append(index)
                cv2.circle(binary_mask, tuple(point), 2, 255, -1)
        return nms_mask

    
    def _form_transf(self, R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t if t.ndim == 1 else t.ravel()
        return T
    
    def make_calibration(self):
        height, width = self.Images['image0'].shape[:2]
        K = np.array([[1500, 0, width/2], [0, 1500, height/2], [0, 0, 1]], dtype=np.float32)
        return K

    def match_images(self, image1, image2, image1_idx, image2_idx):
        if self.keypoints1 is None or self.descriptors1 is None:
            kp1, des1 = self.descriptor.detectAndCompute(image1, None)
            nms_mask = self.non_maximum_suppression(kp1)
            kp1 = np.array(kp1)[nms_mask]
            des1 = np.array(des1)[nms_mask]
        else:
            kp1 = self.keypoints1
            des1 = self.descriptors1

        # Compute keypoints and descriptors for the second image
        kp2, des2 = self.descriptor.detectAndCompute(image2, None)
        nms_mask = self.non_maximum_suppression(kp2)
        kp2 = np.array(kp2)[nms_mask]
        des2 = np.array(des2)[nms_mask]

        # Update stored keypoints and descriptors
        self.keypoints1 = kp2
        self.descriptors1 = des2

        # Find matches using KNN
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if self.show_matches:
            # Create a concatenated image for side-by-side display
            match_image = np.hstack((image1, image2))
            image1_width = image1.shape[1]

            # Convert BGR to RGB for Matplotlib display
            match_image_rgb = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)

            # Plot the concatenated image
            plt.figure(figsize=(12, 6))
            plt.imshow(match_image_rgb)
            plt.axis('off')

            # Plot matches
            for match in good_matches:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx

                # Get the matching keypoints
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt
                x2 += image1_width

                # Draw circles on keypoints
                plt.plot(x1, y1, 'ro', markersize=5)
                plt.plot(x2, y2, 'ro', markersize=5)

                # Draw a line connecting the matched keypoints
                plt.plot([x1, x2], [y1, y2], 'y-', linewidth=0.5)

            plt.show()

        # Extract the matched keypoints' coordinates
        q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        if self.show_keypoints:
            # Plot keypoints on the second image
            plt.figure(figsize=(8, 6))
            image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            plt.imshow(image2_rgb)
            plt.axis('off')

            # Extract keypoint coordinates
            keypoints = np.array([kp.pt for kp in kp2])
            plt.scatter(keypoints[:, 0], keypoints[:, 1], s=15, c='r', marker='o')

            plt.show()

        return q1, q2

    
    def get_Refined_Essential_matrix(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0)
        q1 = q1[mask.ravel() == 1]
        q2 = q2[mask.ravel() == 1]
        return E, q1, q2, mask

    def get_rotation_and_translation(self, E, q1, q2, mask):
        _, R, t, mask = cv2.recoverPose(E, q1, q2, self.K)
        return q1, q2, R, t

    def get_projection_matrix(self, T):
        return self.K @ np.linalg.inv(T)[:3, :]

    def triangulate_points(self, q1, q2, P1, P2):
        points_4d = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        mask = points_3d[2, :] > 0
        points_3d = points_3d[:, mask]
        q1 = q1[mask]
        q2 = q2[mask]
        return points_3d.T, q1, q2

    def Pose_using_PnP(self, Q, q):
        success, rvec, t, inliers = cv2.solvePnPRansac(Q, q, self.K, None)
        if success:
            R, _ = cv2.Rodrigues(rvec)
        T_c_w = self._form_transf(R, t)
        return T_c_w

    def plot_match_points(self, left_image, right_image, left_points, right_points):
        plt.cla()
        match_image = np.hstack((left_image, right_image))
        left_image_width = left_image.shape[1]
        for left_point, right_point in zip(left_points.astype(int), right_points.astype(int)):
            left_match_point = tuple(left_point)
            right_match_point = tuple(right_point + np.array([left_image_width, 0]))
            cv2.circle(match_image, left_match_point, 5, (0, 255, 0), -1)
            cv2.circle(match_image, right_match_point, 5, (0, 255, 0), -1)
            cv2.line(match_image, left_match_point, right_match_point, (0, 255, 0), 1)
        plt.imshow(match_image)
        plt.pause(1)


        
    def calculate_point_colors(self, cloud):
        cloud_colors = []
        for point in cloud:
            r, g, b = 0, 0, 0
            for key in point.origin.keys():
                image = self.Images[f'image{key}_color']
                point2d = point.origin[key]
                r += image[int(point2d[1]), int(point2d[0]), 0]
                g += image[int(point2d[1]), int(point2d[0]), 1]
                b += image[int(point2d[1]), int(point2d[0]), 2]
            
            # Average the color by dividing by the number of views    
            r /= len(point.origin.keys())
            g /= len(point.origin.keys())
            b /= len(point.origin.keys())
            
            # Append the color to the list
            cloud_colors.append([r, g, b])
        
        cloud_colors = np.array(cloud_colors)
        return cloud_colors
    
    def initialize_pointcloud(self):
        print("Processing initial images (image 0 and image 1)...")

        # Load the first two images
        image1, image2 = self.Images['image0'], self.Images['image1']
        q1, q2 = self.match_images(image1, image2, 0, 1)

        E, q1, q2, mask = self.get_Refined_Essential_matrix(q1, q2)
        print(f"Points after essential matrix refinement: {q1.shape}, {q2.shape}")

        q1, q2, R, t = self.get_rotation_and_translation(E, q1, q2, mask)
        print(f"Points after translation inliers: {q1.shape}, {q2.shape}")

        # Optionally plot matched points for visualization
        if self.plot_matching:
            self.plot_match_points(image1, image2, q1, q2)

        # Compute the relative transformation matrix between cameras
        T_2_1 = self._form_transf(R, t)
        self.camera_poses['T_w_0'] = np.eye(4)  # First camera pose (identity matrix)
        self.camera_poses['T_w_1'] = np.linalg.inv(T_2_1)  # Second camera pose

        # Compute projection matrices for triangulation
        P1 = self.get_projection_matrix(self.camera_poses['T_w_0'])
        P2 = self.get_projection_matrix(self.camera_poses['T_w_1'])

        # Triangulate 3D points from matched features
        points_3D, q1, q2 = self.triangulate_points(q1, q2, P1, P2)
        print(f"Triangulated 3D points: {points_3D.shape}, refined matches: {q1.shape}, {q2.shape}")

        # Initialize the point cloud with triangulated points
        self.pointcloud = PointCloud_Capture(points_3D, 0, q1, q2)
        self.prev_keypoints = q2

    
    def update_pointcloud_with_PnP(self):
        # Loop over the remaining images
        for i in range(1, self.N-1):
            print(f"Working on image {i} and {i+1}")
            image1 = self.Images[f'image{i}']
            image2 = self.Images[f'image{i+1}']
            
            # Match the images
            q1, q2 = self.match_images(image1, image2, i, i+1)
            
            # Find the essential matrix and refine the matches
            E, q1, q2, mask = self.get_Refined_Essential_matrix(q1, q2)
            print(f"Points after essential matrix refinement: {q1.shape}, {q2.shape}")
            
            if self.plot_matching:
                self.plot_match_points(self.Images[f'image{i}'], self.Images[f'image{i+1}'], q1, q2)
            
            # Scan the pointcloud to track 3d points from previous image
            print("Comparison btwn prev and current keypoint sizes: ", self.prev_keypoints.shape, q1.shape)
            matched_2d_image1, matched_3d, matched_indices = self.pointcloud.match_points(i, self.prev_keypoints, q1)
            print("Matched points of 2d and 3d: ", matched_2d_image1.shape, matched_3d.shape)
            
            # Find matching 2d points in the second image to calculate poses using PnP
            matched_2d_image2 = q2[matched_indices]
                        
            # Compute camera pose using PnP and update the camera pose list
            T_2_w = self.Pose_using_PnP(matched_3d, matched_2d_image2)
            self.camera_poses[f'T_w_{i+1}'] = np.linalg.inv(T_2_w)
            
            # Compute projection matrix
            P1 = self.get_projection_matrix(self.camera_poses[f'T_w_{i}'])
            P2 = self.get_projection_matrix(self.camera_poses[f'T_w_{i+1}'])
            
            # Triangulate points
            points3D, q1, q2 = self.triangulate_points(q1, q2, P1, P2)
            print("After triangulation 3d Points: ", points3D.shape, q1.shape, q2.shape)
            
            # Update pointcloud
            self.pointcloud.add_or_update(i, points3D, q1, q2)
            self.prev_keypoints = q2

    
    def bundle_adjustment(self):
        # Create the shorthand variables
        L = gtsam.symbol_shorthand.L
        X = gtsam.symbol_shorthand.X
        
        # Create the camera matrix factor and the calibration matrix key
        K = gtsam.Cal3_S2(self.K[0, 0], self.K[1, 1], 0.0, self.K[0, 2], self.K[1, 2])
        
        # Define the camera observation noise model
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0)  # one pixel in u and v
        
        # Create the factor graph
        graph = gtsam.NonlinearFactorGraph()
        

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        factor = gtsam.PriorFactorPose3(X(0), gtsam.Pose3(self.camera_poses['T_w_0']), pose_noise)
        graph.push_back(factor)
        
        # Loop over the points
        for j, point in enumerate(self.pointcloud.cloud):
            keys = list(point.origin.keys())
            for i in keys:
                measurement = point.origin[i]
                factor = gtsam.GenericProjectionFactorCal3_S2(measurement, measurement_noise, X(int(i)), L(j), K)
                graph.push_back(factor)    
        
        # Add the prior on the position of the first landmark
        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        factor = gtsam.PriorFactorPoint3(L(0), gtsam.Point3(self.pointcloud.cloud[0].coords), point_noise)
        graph.push_back(factor)
        
        # Intialize the estimates
        initial_estimate = gtsam.Values()
        # initial_estimate.insert(K_key, K)
        for i in range(self.N):
            initial_estimate.insert(X(i), gtsam.Pose3(self.camera_poses[f'T_w_{i}']))
            
        for j, point in enumerate(self.pointcloud.cloud):
            initial_estimate.insert(L(j), gtsam.Point3(point.coords))    

        # Optimize the graph
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        print("Final error: ", graph.error(result))
        print("--------------------------------------------")

        
        # Update the camera poses
        for i in range(self.N):
            self.camera_poses[f'T_w_{i}'] = result.atPose3(X(i)).matrix()
            
        # Update the pointcloud
        for j, point in enumerate(self.pointcloud.cloud):
            self.pointcloud.cloud[j].coords = result.atPoint3(L(j))

    def optimize_K(self):
        # Create the shorthand variables
        L = gtsam.symbol_shorthand.L
        X = gtsam.symbol_shorthand.X
    
        # Create the camera matrix factor and the calibration matrix key
        K = gtsam.Cal3_S2(self.K[0, 0], self.K[1, 1], 0.0, self.K[0, 2], self.K[1, 2])
        K_key = 20000
        
        # Define the camera observation noise model
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
        
        # Create the factor graph
        graph = gtsam.NonlinearFactorGraph()

        for i in range(self.N):
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
            factor = gtsam.PriorFactorPose3(X(i), gtsam.Pose3(self.camera_poses[f'T_w_{i}']), pose_noise)
            graph.push_back(factor)
        
        # Loop over the points
        for j, point in enumerate(self.pointcloud.cloud):
            keys = list(point.origin.keys())
            for i in keys:
                measurement = point.origin[i]
                factor = gtsam.GeneralSFMFactor2Cal3_S2(measurement, measurement_noise, X(int(i)), L(j), K_key)
                graph.push_back(factor)
                

        # Add the prior on the position of the first landmark
        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        factor = gtsam.PriorFactorPoint3(L(0), gtsam.Point3(self.pointcloud.cloud[0].coords), point_noise)
        graph.push_back(factor)
        
        # Intialize the estimates
        initial_estimate = gtsam.Values()
        initial_estimate.insert(K_key, K)
        for i in range(self.N):
            initial_estimate.insert(X(i), gtsam.Pose3(self.camera_poses[f'T_w_{i}']))
            
        for j, point in enumerate(self.pointcloud.cloud):
            initial_estimate.insert(L(j), gtsam.Point3(point.coords))      
        
        # Plot the factor graph
        # self.plot_factor_graph(graph, initial_estimate, X, L, "Factor Graph before optimization")
        
        # Optimize the graph
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        print("Final error: ", graph.error(result))
        
        print("--------------------------------------------")
        print("Optimized Calibration \n{}".format(result.atCal3_S2(K_key)))

        
        # Update the camera poses
        for i in range(self.N):
            self.camera_poses[f'T_w_{i}'] = result.atPose3(X(i)).matrix()
            
        # Update the pointcloud
        for j, point in enumerate(self.pointcloud.cloud):
            self.pointcloud.cloud[j].coords = result.atPoint3(L(j))

    def main(self):
        # Read the images
        print("Reading images...")
        self.Images, self.N = self.read_images()
        
        # Make the calibration matrix
        self.K = self.make_calibration()
        
        # Intialize the point cloud
        print("Intializing pointcloud...")
        self.initialize_pointcloud()
        
        print("Starting main loop...")
        # Match next Images with pointcloud and update the pointcloud
        self.update_pointcloud_with_PnP()
        
        # Calculate the colors of the points
        cloud_colors = self.calculate_point_colors(self.pointcloud.cloud)

        # Plot the point cloud
        Visualizer.display_pointcloud(self.pointcloud.extract_coordinates(), cloud_colors)

        Visualizer.display_scene(
            self.camera_poses,
            show_cameras=True,
            show_pointcloud=False,
            title = "Camera Poses Before BA"
            )
        
        # Plot the camera poses and the pointcloud
        Visualizer.display_scene(
            self.camera_poses, 
            self.pointcloud.extract_coordinates(), 
            cloud_colors, 
            show_pointcloud=True,
            title = "camera Pose and Pointcloud Before BA"
            )
        
        if self.use_BA:
            print("Starting Bundle Adjustment...")
            self.bundle_adjustment()

            # Visualize updated camera poses and point cloud after bundle adjustment
            Visualizer.display_scene(
                self.camera_poses, 
                self.pointcloud.extract_coordinates(), 
                cloud_colors, 
                show_pointcloud=True,
                title = "Final Point cloud and camera Poses together after BA"
            )

            Visualizer.display_scene(
                self.camera_poses,
                show_cameras=True,
                show_pointcloud=False,
                title = "camera Pose After BA"
            
            )

if __name__=="__main__":
    data_path = "/home/abhinav/Desktop/HW4/buddha_images"
    sfm = SFM_LOOP(data_path, descriptor_name="SIFT", num_features=1000, matcher="BFMatcher", show_matches=True, 
              use_BA=True, plot_matching=False)
    sfm.main()