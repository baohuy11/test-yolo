import cv2
import torch
import numpy as np
# Removed: from ultralytics.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors # Use Annotator from utils
import supervision as sv
from shapely.geometry import Point, Polygon

# Define colors (can be customized)
REGION_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 128, 0)
]
OBJECT_COLORS = [ # Renamed from vehicle_color for clarity
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 128, 0)
]

class MultiRegionCounter:
    # Removed inheritance from BaseSolution
    def __init__(self, regions, names, model, tracker_config, reg_pts=None, classes_names=None, line_thickness=2):
        # Removed super().__init__(**kwargs)
        # Accept model and tracker_config directly

        self.model = model # Assign the passed model instance
        self.tracker_config = tracker_config # Store tracker config path
        # Note: Tracker initialization might need to happen here or be handled differently
        # depending on how ultralytics expects tracker usage without BaseSolution.
        # For now, we assume self.tracker will be available later or needs explicit setup.
        # Placeholder for tracker instance if needed explicitly:
        # self.tracker = self._initialize_tracker(tracker_config)

        # Use reg_pts if provided, otherwise handle regions directly
        self.regions = regions # Expect regions to be passed directly
        if not self.regions:
             raise ValueError("Regions must be provided either via 'regions' argument or configuration.")

        self.names = names # Class names from the model

        # Prepare separate counters and sets for each region
        self.in_count = [0] * len(self.regions)
        self.out_count = [0] * len(self.regions)
        self.counted_ids = [set() for _ in range(len(self.regions))]

        self.regions_initialized = False
        self.polygons = [] # Store Shapely Polygons
        self.track_history = {} # Dictionary to store previous centroid for each track_id

        # Drawing settings
        self.line_thickness = line_thickness

    # Placeholder for tracker initialization if needed
    # def _initialize_tracker(self, config_path):
    #     # Logic to load the tracker based on the config file
    #     # This depends on the specific tracker library/API
    #     print(f"Initializing tracker with config: {config_path}")
    #     # Example: return SomeTrackerClass(config_path)
    #     # For now, we'll rely on the model's built-in tracking or handle it later
    #     # Returning None or raising NotImplementedError might be appropriate
    #     # Let's assume the model handles tracking for now based on the config.
    #     # Accessing self.tracker later might require adjustments.
    #     return None # Placeholder

    def initialize_region_geometry(self):
        """Initializes Shapely Polygons for each region."""
        self.Point = Point # Assign class directly
        self.Polygon = Polygon # Assign class directly
        self.polygons = [self.Polygon(region_points) for region_points in self.regions]
        self.regions_initialized = True
        print(f"Initialized {len(self.polygons)} region polygons.")

    def count_objects(self, region_idx, region_polygon, current_centroid, track_id, prev_centroid):
        """Counts objects entering/leaving a specific region."""
        if prev_centroid is None or track_id in self.counted_ids[region_idx]:
            return # Skip if no previous position or already counted in this region

        # Check if the object is currently inside the region polygon
        is_inside = region_polygon.contains(self.Point(current_centroid))
        # Check if the object was previously inside the region polygon
        was_inside = region_polygon.contains(self.Point(prev_centroid))

        # More robust in/out logic: Check transition across boundary
        if is_inside and not was_inside: # Entered the region
             # Simple direction check (can be refined)
            region_points = list(region_polygon.exterior.coords)
            xs = [p[0] for p in region_points]
            ys = [p[1] for p in region_points]
            region_width = max(xs) - min(xs)
            region_height = max(ys) - min(ys)

            going_in = False # Default assumption
            # Basic direction heuristic (adjust based on camera angle/region setup)
            if region_width < region_height: # Primarily vertical region
                if current_centroid[0] > prev_centroid[0]: # Moving right (example 'in')
                    going_in = True
            else: # Primarily horizontal region
                 if current_centroid[1] > prev_centroid[1]: # Moving down (example 'in')
                    going_in = True

            if going_in:
                self.in_count[region_idx] += 1
                print(f"Track {track_id} entered region {region_idx} (IN). Total IN: {self.in_count[region_idx]}")
            else:
                self.out_count[region_idx] += 1
                print(f"Track {track_id} entered region {region_idx} (OUT). Total OUT: {self.out_count[region_idx]}")

            self.counted_ids[region_idx].add(track_id)

        # Optional: Handle leaving logic (if needed)
        # elif was_inside and not is_inside: # Left the region
        #     print(f"Track {track_id} left region {region_idx}")
        #     # Reset counted status if object can re-enter and be counted again
        #     if track_id in self.counted_ids[region_idx]:
        #          self.counted_ids[region_idx].remove(track_id)


    def display_count(self, plot_img):
        """Displays the in/out counts on the image for each region."""
        # Use a single Annotator instance passed or created outside if needed multiple times,
        # or create one here just for text.
        annotator = Annotator(plot_img, line_width=self.line_thickness, example=str(self.names))
        for i, region_points in enumerate(self.regions):
            # Removed incorrect annotator.polygon call from here
            xs = [p[0] for p in region_points]
            ys = [p[1] for p in region_points]
            center_x = int(sum(xs) / len(xs))
            center_y = int(sum(ys) / len(ys))

            text_str = f"In:{self.in_count[i]} Out:{self.out_count[i]}"

            # Use Annotator for drawing text (Ensuring box_style is removed)
            annotator.text((center_x, center_y), text_str, txt_color=(255, 255, 255))


    # Note: process_frame is no longer used as logic moved to main loop
    # def process_frame(self, frame):
    #     ... (keep the old code commented or remove if desired) ...
        """Processes a single frame for object detection, tracking, and counting."""
        if not self.regions_initialized:
            self.initialize_region_geometry()

        # Run detection and tracking (assuming BaseSolution provides self.results)
        # This part depends heavily on how BaseSolution integrates the model and tracker
        # Let's assume self.model(frame, ...) returns results and self.track(results) updates tracks
        # The following is a plausible structure based on ultralytics examples:

        results = self.model(frame, verbose=False)[0] # Get results for the first image
        if results.boxes.id is None:
             print("Tracker not initialized or no tracks found.")
             tracks = []
        else:
            # Assuming results have boxes with xyxy, conf, cls, id
            # Convert results to a format suitable for store_tracking_history and iteration
            # This might need adjustment based on the exact format from your model/tracker
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.float().cpu().tolist()

            tracks = list(zip(boxes_xyxy, track_ids, classes, confidences)) # Combine info

        annotated_frame = results.plot() # Use default plotting from results
        annotator = Annotator(annotated_frame, line_width=self.line_thickness, example=str(self.names))

        # Draw regions
        for i, region_points in enumerate(self.regions):
            color_idx = i % len(REGION_COLORS)
            color = REGION_COLORS[color_idx]
            annotator.polygon(np.array(region_points), color=color, thickness=self.line_thickness * 2)

            # Optional: Draw filled polygon using supervision
            # try:
            #     b, g, r = color
            #     sv_color = sv.Color(r=r, g=g, b=b)
            #     annotated_frame = sv.draw_polygon(
            #         scene=annotated_frame,
            #         polygon=np.array(region_points), # Use polygon instead of points
            #         color=sv_color,
            #         # opacity=0.5 # draw_polygon doesn't have opacity, use draw_filled_polygon
            #     )
            #     annotated_frame = sv.draw_filled_polygon(
            #          scene=annotated_frame,
            #          polygon=np.array(region_points),
            #          color=sv_color.with_opacity(0.3) # Example opacity
            #     )
            # except Exception as e:
            #      print(f"Error drawing SV polygon: {e}")


        # Process detected tracks
        for box, track_id, cls_id, conf in tracks:
            # Draw bounding box and label (already done by results.plot(), but can customize here)
            # label = f"{self.names[cls_id]} {track_id} ({conf:.2f})"
            # color_idx = track_id % len(OBJECT_COLORS)
            # annotator.box_label(box, label, color=OBJECT_COLORS[color_idx])

            # Store history (Tracker might handle this implicitly when model is called with tracker config)
            # self.store_tracking_history(track_id, box)

            # Get current and previous positions for counting
            current_centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

            # --- Accessing Previous Position ---
            # How prev_position is obtained depends on the tracker implementation.
            # If the model handles tracking internally (e.g., YOLO(...).track(...)),
            # the results object might contain history or the tracker instance might be
            # accessible via the model object. This needs verification based on ultralytics API.
            # Placeholder: Assume results or model object provides access.
            # Let's try accessing tracker history if available on the model or results.
            # This is speculative and might need debugging/adjustment.
            prev_position = None
            # Example: Check if model has tracker attribute and get_prev_position method
            if hasattr(self.model, 'tracker') and hasattr(self.model.tracker, 'get_prev_position'):
                 try:
                     prev_position = self.model.tracker.get_prev_position(track_id)
                 except Exception as e:
                     print(f"Could not get previous position for track {track_id}: {e}")
            elif results.tracker: # Check if tracker info/method is on results
                 # Adapt based on actual API if tracker info is here
                 pass # Add logic if tracker info is found on results

            if prev_position:
                prev_centroid = (int(prev_position[0]), int(prev_position[1])) # Assuming prev_position is (x, y)
                # Iterate through regions to count
                for region_idx, region_polygon in enumerate(self.polygons):
                    self.count_objects(region_idx, region_polygon, current_centroid, track_id, prev_centroid)
            else:
                 # Store initial position if tracker doesn't handle it automatically
                 # This depends on the specific tracker implementation
                 pass

        # Display counts on the frame
        self.display_count(annotated_frame)

        return annotated_frame


# --- Main Execution ---
if __name__ == "__main__":
    video_path = "video.mp4"
    output_path = "output.mp4"
    model_name = "yolov8n.pt" # Example model, adjust as needed
    tracker_config = "botsort.yaml" # Example tracker

    # Define regions as a list of lists of points
    # Extracted from the original dictionary
    region_definitions = [
        [[457, 223], [489, 229], [135, 410], [7, 408]],      # region-1
        [[491, 227], [540, 229], [272, 416], [135, 410]],    # region-2
        [[550, 213], [603, 217], [416, 422], [276, 414]],    # region-3
        [[607, 217], [652, 217], [548, 428], [420, 420]],    # region-4
        [[762, 219], [803, 215], [1014, 430], [886, 441]],   # region-5
        [[805, 211], [839, 207], [1143, 412], [1016, 430]],   # region-6
        [[843, 204], [874, 202], [1250, 402], [1143, 410]],   # region-7
        [[886, 202], [913, 200], [1377, 402], [1248, 400]],   # region-8
    ]

    # --- Video Setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- Video Writer Setup ---
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- Initialize the Counter ---
    # This assumes BaseSolution or its parent handles model loading and tracker setup
    # based on 'model' and 'tracker' kwargs.
    # We need the class names from the model. Load the model temporarily or pass names if known.
    # Example: Load model to get names (replace with actual model loading if needed)
    from ultralytics import YOLO
    temp_model = YOLO(model_name)
    class_names = temp_model.names

    # Pass the initialized model instance and tracker config
    region_counter = MultiRegionCounter(
        regions=region_definitions,
        names=class_names,
        model=temp_model,         # Pass the initialized YOLO model
        tracker_config=tracker_config, # Pass the tracker config path
        line_thickness=2,
    )

    # --- Processing Loop ---
    # Modify model call to include tracking
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Reached end of video or error reading frame.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Process the frame using the counter instance
        # The process_frame method now uses the model instance directly
        # We need to ensure the model call within process_frame performs tracking
        # Let's modify the model call inside process_frame to enable tracking

        # --- Modification needed inside process_frame ---
        # The call `results = self.model(frame, verbose=False)[0]` needs `tracker=self.tracker_config`
        # Let's adjust process_frame directly (can't do easily with replace_in_file here)
        # OR adjust the main loop call:

        # Instead of calling region_counter.process_frame, perform model tracking here
        # and pass the results to a simplified process_frame or handle logic here.
        # Let's try modifying process_frame first. This requires another replace_in_file pass.

        # --- Alternative: Call model.track here and pass results ---
        results = region_counter.model.track(frame, persist=True, tracker=region_counter.tracker_config, verbose=False)[0]

        # Now, adapt process_frame to accept results instead of frame, or refactor logic here.
        # Let's refactor the core logic from process_frame to work with results here.

        if results.boxes.id is None:
             print("Tracker not initialized or no tracks found.")
             tracks = []
             annotated_frame = results.plot() # Plot frame even without tracks
        else:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.float().cpu().tolist()
            tracks = list(zip(boxes_xyxy, track_ids, classes, confidences))

            annotated_frame = results.plot() # Use default plotting from results
            annotator = Annotator(annotated_frame, line_width=region_counter.line_thickness, example=str(region_counter.names))

            # Draw regions using Supervision
            if not region_counter.regions_initialized:
                region_counter.initialize_region_geometry()

            for i, region_points in enumerate(region_counter.regions):
                color_idx = i % len(REGION_COLORS)
                # Get BGR color tuple
                b, g, r = REGION_COLORS[color_idx]
                # Create SV Color using RGB order
                sv_color = sv.Color(r=r, g=g, b=b)

                # Draw filled polygon with opacity using supervision
                try:
                    # Draw filled polygon using supervision (removed opacity)
                    annotated_frame = sv.draw_filled_polygon(
                         scene=annotated_frame,
                         polygon=np.array(region_points),
                         color=sv_color
                    )
                    # Draw polygon outline using supervision
                    annotated_frame = sv.draw_polygon(
                        scene=annotated_frame,
                        polygon=np.array(region_points),
                        color=sv_color,
                        thickness=region_counter.line_thickness # Use defined thickness
                    )
                except Exception as e:
                     print(f"Error drawing SV polygon for region {i}: {e}")


            # Process detected tracks
            for box, track_id, cls_id, conf in tracks:
                current_centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                # --- Get Previous Position from Manual History ---
                prev_centroid = region_counter.track_history.get(track_id)

                # --- Perform Counting ---
                if prev_centroid is not None:
                    # Iterate through regions to count
                    for region_idx, region_polygon in enumerate(region_counter.polygons):
                        region_counter.count_objects(region_idx, region_polygon, current_centroid, track_id, prev_centroid)

                # --- Update History ---
                region_counter.track_history[track_id] = current_centroid


            # Display counts
            region_counter.display_count(annotated_frame)


        # Write the frame to the output video
        video_writer.write(annotated_frame)

        # Display the frame (optional)
        cv2.imshow("Multi-Region Counting", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): # Press 'q' to quit
            print("Exiting...")
            break

    # --- Cleanup ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

    # Print final counts
    print("\n--- Final Counts ---")
    for i in range(len(region_definitions)):
        print(f"Region {i+1}: In={region_counter.in_count[i]}, Out={region_counter.out_count[i]}")
