# Anatomical Knowledge Graph

## Entities

### Skull
- description: The skull is the bony structure that forms the head and protects the brain
- type: bone_structure
- region: head
- intensity_range: [120, 255]

### Mandible
- description: The lower jaw bone
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Maxilla
- description: The upper jaw bone, forms the roof of the mouth
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Zygomatic Bone
- description: The cheekbone
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Frontal Bone
- description: The forehead bone, forms the front of the skull
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Parietal Bone
- description: Forms the sides and roof of the skull
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Temporal Bone
- description: Forms the lower sides of the skull, contains the ear canal
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Occipital Bone
- description: Forms the back and base of the skull
- type: bone
- region: head
- part_of: skull
- intensity_range: [120, 255]

### Brain
- description: The central nervous system organ contained within the skull
- type: soft_tissue
- region: head
- contained_by: skull
- intensity_range: [30, 60]

### Cerebral Cortex
- description: The outer layer of the brain
- type: soft_tissue
- region: head
- part_of: brain
- intensity_range: [35, 50]

### White Matter
- description: The inner tissue of the brain containing nerve fibers
- type: soft_tissue
- region: head
- part_of: brain
- intensity_range: [20, 40]

### Ventricles
- description: Fluid-filled cavities in the brain
- type: fluid
- region: head
- contained_by: brain
- intensity_range: [0, 15]

### Femur
- description: The thigh bone, longest bone in the human body
- type: bone
- region: leg
- intensity_range: [120, 255]

### Tibia
- description: The shin bone
- type: bone
- region: leg
- intensity_range: [120, 255]

### Fibula
- description: The outer bone of the lower leg
- type: bone
- region: leg
- adjacent_to: tibia
- intensity_range: [120, 255]

### Muscle Tissue
- description: Skeletal muscle throughout the body
- type: soft_tissue
- intensity_range: [40, 80]

### Fat Tissue
- description: Adipose tissue
- type: soft_tissue
- intensity_range: [5, 30]

### Blood
- description: Blood in vessels
- type: fluid
- intensity_range: [30, 60]

## Relationships

- mandible is_part_of skull
- maxilla is_part_of skull
- zygomatic_bone is_part_of skull
- frontal_bone is_part_of skull
- parietal_bone is_part_of skull
- temporal_bone is_part_of skull
- occipital_bone is_part_of skull
- brain contained_by skull
- cerebral_cortex is_part_of brain
- white_matter is_part_of brain
- ventricles contained_by brain
- tibia adjacent_to fibula
- mandible connects_to maxilla

## Conventions

### Convention: bone
- color: white
- rgb: [255, 255, 255]
- opacity_range: [0.8, 1.0]
- description: Dense bone structures with high CT values
- intensity_threshold: 100
- rationale: Bones are dense structures with high intensity in CT scans, white provides maximum contrast

### Convention: soft_tissue
- color: red
- rgb: [180, 100, 100]
- opacity_range: [0.3, 0.7]
- description: Muscle and organ tissue
- intensity_range: [30, 80]
- rationale: Mid-range intensities representing soft anatomical structures

### Convention: brain_tissue
- color: pink
- rgb: [200, 150, 150]
- opacity_range: [0.5, 0.8]
- description: Brain tissue (gray and white matter)
- intensity_range: [20, 60]
- rationale: Neural tissue with characteristic CT density

### Convention: fluid
- color: blue
- rgb: [100, 150, 200]
- opacity_range: [0.2, 0.5]
- description: CSF, blood, and other fluids
- intensity_range: [0, 30]
- rationale: Low density fluid-filled spaces

### Convention: fat
- color: yellow
- rgb: [220, 200, 100]
- opacity_range: [0.2, 0.4]
- description: Adipose tissue
- intensity_range: [5, 30]
- rationale: Low-density fatty tissue
