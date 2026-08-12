"""
Vehicle Tracker module leveraging Ultralytics ByteTrack for multi-object tracking.
"""

from collections import defaultdict, deque
from typing import Deque, List, Optional, Dict, Set, Tuple
import cv2
import numpy as np

from backend.detection.yolo_detector import YOLODetector
from backend.models.tracking import TrackedObject, FrameTracks


class TrackerError(Exception):
    """Base exception for tracker errors."""

    pass


class VehicleTracker:
    """ByteTrack wrapper integrating with Ultralytics YOLO for persistent vehicle tracking."""

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.35,
        tracker_type: str = "bytetrack.yaml",
        target_classes: Optional[Dict[int, str]] = None,
        high_recall: bool = False,
        tile_inference_size: int = 640,
        tile_confidence_threshold: float = 0.18,
        tile_grid_size: int = 2,
        tile_interval_frames: int = 5,
        far_field_recall: bool = False,
        far_field_inference_size: int = 1280,
        far_field_confidence_threshold: float = 0.05,
        detection_memory_frames: int = 0,
        class_history_frames: int = 12,
        heavy_vehicle_min_confidence: float = 0.30,
        heavy_vehicle_min_observations: int = 3,
        bus_min_observations: int = 2,
        class_switch_margin: float = 1.20,
        suppress_camera_overlay: bool = True,
        overlay_top_fraction: float = 0.24,
        overlay_left_fraction: float = 0.50,
    ):
        """Args:

        detector: Existing YOLODetector instance, or None to create a new one.
        model_path: Path to YOLO model weights if detector is None.
        confidence_threshold: Confidence score threshold.
        tracker_type: Ultralytics tracker configuration ('bytetrack.yaml' or 'botsort.yaml').
        target_classes: Dict mapping class_id -> class_name for vehicle filtering.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLODetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                target_classes=target_classes,
            )

        self.tracker_type = tracker_type
        self.target_classes = target_classes or self.detector.target_classes
        self.high_recall = high_recall
        self.tile_inference_size = max(320, int(tile_inference_size))
        self.tile_confidence_threshold = min(
            1.0, max(0.01, float(tile_confidence_threshold))
        )
        self.tile_grid_size = min(3, max(2, int(tile_grid_size)))
        self.tile_interval_frames = max(1, int(tile_interval_frames))
        self.far_field_recall = bool(far_field_recall)
        self.far_field_inference_size = max(
            self.tile_inference_size, int(far_field_inference_size)
        )
        self.far_field_confidence_threshold = min(
            1.0, max(0.01, float(far_field_confidence_threshold))
        )
        self.detection_memory_frames = max(0, int(detection_memory_frames))
        self.class_history_frames = max(2, int(class_history_frames))
        self.heavy_vehicle_min_confidence = min(
            1.0, max(0.01, float(heavy_vehicle_min_confidence))
        )
        self.heavy_vehicle_min_observations = max(
            1, int(heavy_vehicle_min_observations)
        )
        self.bus_min_observations = max(1, int(bus_min_observations))
        self.class_switch_margin = max(1.0, float(class_switch_margin))
        self.suppress_camera_overlay = bool(suppress_camera_overlay)
        self.overlay_top_fraction = min(
            0.50, max(0.05, float(overlay_top_fraction))
        )
        self.overlay_left_fraction = min(
            0.80, max(0.10, float(overlay_left_fraction))
        )
        self._next_supplemental_id = 1_000_000
        self._previous_supplemental: Dict[int, TrackedObject] = {}
        self._recent_tracks: Dict[int, Tuple[TrackedObject, int]] = {}
        self._class_history: Dict[int, Deque[Tuple[int, float]]] = {}
        self._resolved_classes: Dict[int, int] = {}
        self._class_last_seen: Dict[int, int] = {}
        self._heavy_class_locks: Dict[int, Tuple[int, int]] = {}

    @staticmethod
    def _iou(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> float:
        """Calculate intersection-over-union for duplicate suppression."""
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _extract_tracked_objects(self, results) -> List[TrackedObject]:
        """Convert one Ultralytics tracking result into project models."""
        tracks: List[TrackedObject] = []
        if not results or len(results) == 0:
            return tracks
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracks

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            track_id = idx + 1
            if hasattr(box, "id") and box.id is not None:
                val = box.id
                if hasattr(val, "item"):
                    track_id = int(val.item())
                elif hasattr(val, "__getitem__") and len(val) > 0:
                    elem = val[0]
                    track_id = int(elem.item()) if hasattr(elem, "item") else int(elem)
                else:
                    try:
                        track_id = int(val)
                    except (ValueError, TypeError):
                        pass
            tracks.append(TrackedObject(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.target_classes[cls_id],
                confidence=round(conf, 4),
                bbox=tuple(round(value, 2) for value in xyxy),
            ))
        return tracks

    def _deduplicate_class_overlaps(
        self,
        objects: List[TrackedObject],
        iou_threshold: float = 0.72,
    ) -> List[TrackedObject]:
        """Collapse car/heavy labels emitted for the exact same vehicle box."""
        consolidated: List[TrackedObject] = []
        for item in objects:
            duplicate_index = next(
                (
                    index for index, existing in enumerate(consolidated)
                    if existing.class_id != item.class_id
                    and self._iou(item.bbox, existing.bbox) >= iou_threshold
                ),
                None,
            )
            if duplicate_index is None:
                consolidated.append(item)
                continue

            existing = consolidated[duplicate_index]
            item_name = self.target_classes.get(item.class_id, "")
            existing_name = self.target_classes.get(existing.class_id, "")
            pair = {item_name, existing_name}
            if "car" in pair and pair.intersection({"bus", "truck"}):
                car = item if item_name == "car" else existing
                heavy = existing if item_name == "car" else item
                # Cars are much more common in this camera domain. A heavy label
                # must be over 40% stronger than the competing car label.
                winner = heavy if heavy.confidence > car.confidence * 1.40 else car
            else:
                winner = item if item.confidence > existing.confidence else existing
            consolidated[duplicate_index] = winner
        return consolidated

    def _filter_camera_overlay_artifacts(
        self,
        objects: List[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> List[TrackedObject]:
        """Discard small detections caused by fixed upper-left CCTV text.

        Camera identifiers such as ``I-5 S 188TH ST`` contain high-contrast
        glyphs that tiled inference can mistake for vehicles. The filter is
        deliberately limited to small boxes fully inside the upper-left overlay
        band, leaving the upper-right roadway and larger real objects untouched.
        """
        if not self.suppress_camera_overlay:
            return objects

        overlay_bottom = frame_height * self.overlay_top_fraction
        overlay_right = frame_width * self.overlay_left_fraction
        max_box_width = frame_width * 0.20
        max_box_height = frame_height * 0.18
        filtered: List[TrackedObject] = []
        for item in objects:
            x1, y1, x2, y2 = item.bbox
            center_x = (x1 + x2) / 2.0
            is_overlay_artifact = (
                center_x <= overlay_right
                and y2 <= overlay_bottom
                and (x2 - x1) <= max_box_width
                and (y2 - y1) <= max_box_height
            )
            if not is_overlay_artifact:
                filtered.append(item)
        return filtered

    def _refine_large_trucks(
        self,
        objects: List[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> Set[int]:
        """Promote car-labelled boxes that are clearly truck-sized for their depth.

        Absolute pixels cannot distinguish a nearby car from a distant truck, so
        each candidate is compared only with vehicles whose bottom edge is at a
        similar vertical position. This captures perspective while requiring
        strong size, height, aspect-ratio, and confidence evidence.
        """
        car_id = next(
            (class_id for class_id, name in self.target_classes.items()
             if name == "car"),
            None,
        )
        truck_id = next(
            (class_id for class_id, name in self.target_classes.items()
             if name == "truck"),
            None,
        )
        if car_id is None or truck_id is None:
            return set()

        reliable = [item for item in objects if item.confidence >= 0.25]
        confirmed: Set[int] = set()
        for item in objects:
            if item.class_id != car_id or item.confidence < 0.30:
                continue
            x1, y1, x2, y2 = item.bbox
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area = width * height
            peers = [
                other for other in reliable
                if other.track_id != item.track_id
                and abs(other.bbox[3] - y2) <= frame_height * 0.14
            ]
            if len(peers) < 2:
                continue
            peer_widths = [max(1.0, peer.bbox[2] - peer.bbox[0]) for peer in peers]
            peer_heights = [max(1.0, peer.bbox[3] - peer.bbox[1]) for peer in peers]
            peer_areas = [w * h for w, h in zip(peer_widths, peer_heights)]
            median_height = float(np.median(peer_heights))
            median_area = float(np.median(peer_areas))

            is_truck_sized = (
                height >= frame_height * 0.08
                and height / width >= 1.25
                and height >= median_height * 1.45
                and area >= median_area * 1.80
                and area <= frame_width * frame_height * 0.22
            )
            if is_truck_sized:
                item.class_id = truck_id
                item.class_name = self.target_classes[truck_id]
                confirmed.add(item.track_id)
        return confirmed

    def _tile_bounds(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Create four overlapping tiles that collectively cover the whole frame."""
        tile_fraction = 0.62 if self.tile_grid_size == 2 else 0.55
        tile_width = min(width, max(1, int(round(width * tile_fraction))))
        tile_height = min(height, max(1, int(round(height * tile_fraction))))
        x_span = width - tile_width
        y_span = height - tile_height
        x_positions = sorted(set(
            round(index * x_span / (self.tile_grid_size - 1))
            for index in range(self.tile_grid_size)
        ))
        y_positions = sorted(set(
            round(index * y_span / (self.tile_grid_size - 1))
            for index in range(self.tile_grid_size)
        ))
        return [
            (x, y, x + tile_width, y + tile_height)
            for y in y_positions
            for x in x_positions
        ]

    def _tile_candidates(self, frame: np.ndarray) -> List[TrackedObject]:
        """Detect small/distant vehicles on overlapping full-road tiles."""
        height, width = frame.shape[:2]
        bounds = self._tile_bounds(width, height)
        tiles = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in bounds]
        results = self.detector.model.predict(
            source=tiles,
            conf=self.tile_confidence_threshold,
            iou=self.detector.iou_threshold,
            imgsz=self.tile_inference_size,
            classes=list(self.target_classes),
            device=self.detector.device,
            verbose=False,
        )

        candidates: List[TrackedObject] = []
        for result, (offset_x, offset_y, _, _) in zip(results or [], bounds):
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.target_classes:
                    continue
                xyxy = box.xyxy[0].tolist()
                bbox = (
                    round(xyxy[0] + offset_x, 2),
                    round(xyxy[1] + offset_y, 2),
                    round(xyxy[2] + offset_x, 2),
                    round(xyxy[3] + offset_y, 2),
                )
                candidates.append(TrackedObject(
                    track_id=-1,
                    class_id=cls_id,
                    class_name=self.target_classes[cls_id],
                    confidence=round(float(box.conf[0].item()), 4),
                    bbox=bbox,
                ))
        return self._filter_camera_overlay_artifacts(
            self._deduplicate_class_overlaps(candidates), width, height
        )

    def _far_field_bounds(
        self, width: int, height: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return perspective-shaped zoom crops for tiny distant vehicles.

        The middle crop follows the narrow left roadway where the missed queue
        appears, while the final shallow crop magnifies the curved horizon.
        This gives small vehicles substantially more model pixels than regular
        square tiling without cropping the nearby-road detector's field of view.
        """
        return [
            (0, int(round(height * 0.10)),
             int(round(width * 0.48)), int(round(height * 0.64))),
            (int(round(width * 0.30)), int(round(height * 0.10)),
             int(round(width * 0.66)), int(round(height * 0.75))),
            (int(round(width * 0.45)), int(round(height * 0.06)),
             width, int(round(height * 0.46))),
        ]

    def _far_field_candidates(self, frame: np.ndarray) -> List[TrackedObject]:
        """Detect low-confidence, distant vehicles on strongly enlarged crops."""
        height, width = frame.shape[:2]
        bounds = self._far_field_bounds(width, height)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        crops = []
        for x1, y1, x2, y2 in bounds:
            crop = frame[y1:y2, x1:x2]
            # The source CCTV clips are low-resolution and low-contrast. Local
            # luminance enhancement separates distant vehicle roofs from road
            # pixels before the crop is enlarged for inference.
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            luminance, channel_a, channel_b = cv2.split(lab)
            enhanced = cv2.merge((clahe.apply(luminance), channel_a, channel_b))
            crops.append(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR))
        results = self.detector.model.predict(
            source=crops,
            conf=self.far_field_confidence_threshold,
            iou=self.detector.iou_threshold,
            imgsz=self.far_field_inference_size,
            classes=list(self.target_classes),
            device=self.detector.device,
            verbose=False,
        )

        candidates: List[TrackedObject] = []
        for result, (offset_x, offset_y, _, _) in zip(results or [], bounds):
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.target_classes:
                    continue
                xyxy = box.xyxy[0].tolist()
                candidates.append(TrackedObject(
                    track_id=-1,
                    class_id=cls_id,
                    class_name=self.target_classes[cls_id],
                    confidence=round(float(box.conf[0].item()), 4),
                    bbox=(
                        round(xyxy[0] + offset_x, 2),
                        round(xyxy[1] + offset_y, 2),
                        round(xyxy[2] + offset_x, 2),
                        round(xyxy[3] + offset_y, 2),
                    ),
                ))
        return self._filter_camera_overlay_artifacts(
            self._deduplicate_class_overlaps(candidates), width, height
        )

    def _merge_supplemental(
        self, tracked: List[TrackedObject], candidates: List[TrackedObject]
    ) -> List[TrackedObject]:
        """Merge zoomed-tile detections and preserve their short-term IDs.

        Tile inference often has enough vehicle detail to distinguish a truck
        from a car when the full-frame inference cannot. An overlapping tile
        result is therefore also classification evidence, not merely a duplicate.
        """
        accepted: List[TrackedObject] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            duplicate = next(
                (
                    item for item in tracked
                    if self._iou(candidate.bbox, item.bbox) >= 0.40
                ),
                None,
            )
            if duplicate is not None:
                # A tiled crop is useful for finding missed vehicles, but a
                # single crop classification must never relabel a tracked car.
                continue
            if any(self._iou(candidate.bbox, item.bbox) >= 0.45 for item in accepted):
                continue

            best_id = None
            best_iou = 0.0
            for track_id, previous in self._previous_supplemental.items():
                overlap = self._iou(candidate.bbox, previous.bbox)
                if overlap >= 0.20 and overlap > best_iou:
                    best_id, best_iou = track_id, overlap
            if best_id is None:
                best_id = self._next_supplemental_id
                self._next_supplemental_id += 1
            candidate.track_id = best_id
            accepted.append(candidate)

        self._previous_supplemental = {item.track_id: item for item in accepted}
        return tracked + accepted

    def _carry_supplemental(
        self, tracked: List[TrackedObject]
    ) -> List[TrackedObject]:
        """Carry tiled detections between scans while suppressing full-frame matches."""
        carried = [
            previous
            for previous in self._previous_supplemental.values()
            if not any(
                self._iou(previous.bbox, current.bbox) >= 0.35
                for current in tracked
            )
        ]
        return tracked + carried

    def _stabilize_tracks(
        self, current: List[TrackedObject], frame_index: int
    ) -> List[TrackedObject]:
        """Keep briefly missed vehicles visible without counting overlapping ghosts."""
        if self.detection_memory_frames <= 0:
            return current

        current_ids = {item.track_id for item in current}
        # If a tracker assigns a new ID to nearly the same box, retire the old ID
        # so short-term memory does not double count that vehicle.
        for item in current:
            for old_id, (old_item, _) in list(self._recent_tracks.items()):
                if old_id != item.track_id and self._iou(item.bbox, old_item.bbox) >= 0.60:
                    self._recent_tracks.pop(old_id, None)
            self._recent_tracks[item.track_id] = (item, frame_index)

        stabilized = list(current)
        for track_id, (previous, last_seen) in list(self._recent_tracks.items()):
            if frame_index - last_seen > self.detection_memory_frames:
                self._recent_tracks.pop(track_id, None)
                continue
            if track_id in current_ids:
                continue
            if any(self._iou(previous.bbox, item.bbox) >= 0.35 for item in stabilized):
                continue
            stabilized.append(previous)
        return stabilized

    def _stabilize_classes(
        self,
        current: List[TrackedObject],
        observed_ids: Set[int],
        frame_index: int,
        geometry_heavy_ids: Optional[Set[int]] = None,
    ) -> List[TrackedObject]:
        """Resolve car/bus/truck flicker using confidence-weighted track history.

        Tiny cars are commonly assigned a low-confidence bus or truck label on
        enlarged tiles. A heavy-vehicle label therefore needs repeated strong
        evidence. The object is still retained and counted while its subtype is
        being confirmed.
        """
        car_id = next(
            (class_id for class_id, name in self.target_classes.items()
             if name == "car"),
            None,
        )
        heavy_ids = {
            class_id for class_id, name in self.target_classes.items()
            if name in {"bus", "truck"}
        }
        geometry_heavy_ids = geometry_heavy_ids or set()

        for item in current:
            track_id = item.track_id
            if track_id in observed_ids:
                history = self._class_history.setdefault(
                    track_id, deque(maxlen=self.class_history_frames)
                )
                history.append((item.class_id, float(item.confidence)))
                self._class_last_seen[track_id] = frame_index

                scores: Dict[int, float] = defaultdict(float)
                for class_id, confidence in history:
                    class_name = self.target_classes.get(class_id, "")
                    prior_weight = {
                        "car": 1.20,
                        "motorcycle": 1.0,
                        "truck": 0.75,
                        "bus": 0.70,
                    }.get(class_name, 1.0)
                    scores[class_id] += confidence * prior_weight

                candidate = max(scores, key=scores.get)
                confirmed_heavy = False
                for heavy_id in heavy_ids:
                    required = (
                        self.bus_min_observations
                        if self.target_classes.get(heavy_id) == "bus"
                        else self.heavy_vehicle_min_observations
                    )
                    recent = list(history)[-required:]
                    if (
                        len(recent) == required
                        and all(
                            class_id == heavy_id
                            and confidence >= self.heavy_vehicle_min_confidence
                            for class_id, confidence in recent
                        )
                    ):
                        candidate = heavy_id
                        confirmed_heavy = True
                        break

                if track_id in geometry_heavy_ids and item.class_id in heavy_ids:
                    candidate = item.class_id
                    confirmed_heavy = True
                    # Keep a strong geometry decision stable across short box
                    # jitter or a momentary partial occlusion in later frames.
                    self._heavy_class_locks[track_id] = (
                        item.class_id,
                        max(3, self.class_history_frames // 2),
                    )

                locked = self._heavy_class_locks.get(track_id)
                if locked is not None:
                    locked_class, frames_left = locked
                    candidate = locked_class
                    confirmed_heavy = True
                    if frames_left <= 1:
                        self._heavy_class_locks.pop(track_id, None)
                    else:
                        self._heavy_class_locks[track_id] = (
                            locked_class, frames_left - 1
                        )

                previous = self._resolved_classes.get(track_id)
                if (
                    previous is not None
                    and candidate != previous
                    and not confirmed_heavy
                ):
                    previous_score = scores.get(previous, 0.0)
                    if scores[candidate] < previous_score * self.class_switch_margin:
                        candidate = previous

                if (
                    candidate in heavy_ids
                    and not confirmed_heavy
                ):
                    candidate = (
                        previous
                        if previous is not None
                        else car_id if car_id is not None else item.class_id
                    )

                self._resolved_classes[track_id] = candidate

            resolved = self._resolved_classes.get(track_id)
            if resolved is not None and resolved in self.target_classes:
                item.class_id = resolved
                item.class_name = self.target_classes[resolved]

        expiry = max(
            self.class_history_frames * 2,
            self.detection_memory_frames * 2,
        )
        for track_id, last_seen in list(self._class_last_seen.items()):
            if frame_index - last_seen > expiry:
                self._class_last_seen.pop(track_id, None)
                self._class_history.pop(track_id, None)
                self._resolved_classes.pop(track_id, None)
                self._heavy_class_locks.pop(track_id, None)

        return current

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> FrameTracks:
        """Process a single frame sequentially and return persistent object tracks.

        Args:
            frame: OpenCV BGR image matrix.
            frame_index: Frame sequence index.
            timestamp: Frame timestamp in seconds.

        Returns:
            FrameTracks dataclass containing list of TrackedObject instances.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame provided for vehicle tracking.")

        if self.detector.model is None:
            raise TrackerError("Underlying YOLO model is not initialized.")

        # Run ByteTrack multi-object tracking via Ultralytics
        results = self.detector.model.track(
            source=frame,
            conf=self.detector.confidence_threshold,
            iou=self.detector.iou_threshold,
            imgsz=self.detector.inference_size,
            classes=list(self.target_classes),
            persist=True,  # Maintains track states across consecutive frames
            tracker=self.tracker_type,
            device=self.detector.device,
            verbose=False,
        )

        frame_height, frame_width = frame.shape[:2]
        tracks = self._filter_camera_overlay_artifacts(
            self._deduplicate_class_overlaps(
                self._extract_tracked_objects(results)
            ),
            frame_width,
            frame_height,
        )
        observed_ids = {item.track_id for item in tracks}
        if self.high_recall:
            should_scan_tiles = (
                not self._previous_supplemental
                or frame_index % self.tile_interval_frames == 0
            )
            if should_scan_tiles:
                candidates = self._tile_candidates(frame)
                if self.far_field_recall:
                    candidates.extend(self._far_field_candidates(frame))
                tracks = self._merge_supplemental(
                    tracks, candidates
                )
                observed_ids.update(item.track_id for item in tracks)
            else:
                tracks = self._carry_supplemental(tracks)
        geometry_heavy_ids = self._refine_large_trucks(
            tracks, frame_width, frame_height
        )
        tracks = self._stabilize_classes(
            tracks,
            observed_ids,
            frame_index,
            geometry_heavy_ids=geometry_heavy_ids,
        )
        tracks = self._stabilize_tracks(tracks, frame_index)

        return FrameTracks(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=tracks,
        )
