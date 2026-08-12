# Traffic vehicle annotation requirements

## Classes

| ID | Class | Include |
|---:|---|---|
| 0 | car | Passenger cars, taxis, SUVs, pickups, and light passenger/cargo vans. |
| 1 | motorcycle | Motorcycles, scooters, and mopeds. Do not label the rider separately. |
| 2 | bus | City, school, shuttle, coach, and clearly passenger-service buses. |
| 3 | truck | Freight lorries, semitrailers, tankers, box trucks, dump trucks, and heavy commercial trucks. |

Use the definitions consistently. In this dataset, pickups and light vans are
`car`; large freight vehicles are `truck`.

## Bounding boxes

- Draw one tight box around every clearly identifiable target vehicle.
- Include the visible vehicle body and wheels, with as little road/background
  as practical.
- For a partially occluded vehicle, box the visible extent; do not guess the
  hidden boundary.
- Clip boxes at the image edge for truncated vehicles.
- Label small and distant vehicles when the class is still identifiable. Zoom
  in while annotating. Do not guess a class for an indistinguishable few-pixel
  object.
- Label each distinct vehicle separately in queues and overlaps.
- Do not label people, bicycles, trains, reflections, shadows, signs, or images
  of vehicles on billboards.
- Review dense frames at high zoom for missed objects.

## YOLO label format

Each image must have a matching `.txt` file under the corresponding `labels`
split. Each object is one line:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. Example:

```text
3 0.625000 0.540000 0.180000 0.220000
```

Create an empty `.txt` file only after confirming that an image contains no
target vehicles. A missing file means annotation is not complete.

## Coverage and quality review

- Include small/distant, near-camera, partially occluded, front/rear/side, and
  different-angle examples.
- Include varying illumination, weather, traffic density, road direction, and
  camera viewpoint.
- Seek additional source videos containing motorcycles, buses, and trucks;
  never duplicate frames merely to make class counts look balanced.
- Before training, verify every image has a reviewed label file and run a
  visual label audit on all validation and test images.
