## Rendering Conventions

- when displaying bones, make them white
- bones are typically displayed in white because they are dense structures with high intensity values in CT scans, and white provides maximum contrast against dark backgrounds for medical visualization
- when rendering bone structures, use a high threshold value (typically above intensity 100) to isolate bones from soft tissue

## Reference Images

- images/skull_front.jpg shows the reference image for a skull from a front view
- images/skull_side.jpg shows the reference image from for a skull from a side view
- show the front view first

## Camera Positioning Strategy

- use spherical coordinates when rendering with elevation (vertical angle) and azimuth (horizontal angle) to position the camera
- elevation controls how high or low the camera is: 0 degrees is eye level, 90 degrees is top view, -90 degrees is bottom view
- azimuth controls rotation around the subject: 0 degrees is front, 90 degrees is right side, 180 degrees is back, 270 degrees is left side
- when the angle doesn't match the reference, adjust elevation and azimuth incrementally until finding the right viewpoint
- if this is not the right angle, change elevation by 15 degrees and azimuth by 15 degrees to search for better matches
- camera distance controls zoom: higher distance values (2.0-5.0) zoom out, lower values (1.0-1.5) zoom in

## Framing and Cropping Rules

- when displaying an image, maximize the use of the space but do not crop
- an image is cropped if part of the anatomy is cut off at the edges
- a properly framed image should have background color visible on all sides of the anatomy
- when rendering, start zoomed out (distance 5.0 or higher) and gradually zoom in until the anatomy fills most of the frame without being cropped
- if the anatomy appears cropped, increase camera distance by 0.5 increments until the full structure is visible
- aim for the anatomy to occupy 70-90% of the image space while remaining fully visible

## Workflow for Matching Reference Images

- when asked for a specific view (front, side, top), first retrieve the reference image from the knowledge graph
- compare the initial render with the reference image to check if angles match
- if angles don't match, use the automatic angle finder to test different elevation and azimuth combinations
- adjust camera position iteratively: change elevation and azimuth by 15-30 degree increments based on the mismatch
- once the correct angle is found, fine-tune the camera distance for proper framing (not cropped, well-centered)