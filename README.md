# Machine Learning for 3D Geometry

## Outline

- Shape Classification and Segmentation
- Surface Representation and Alignment
- 3D Shape Reconstruction
- References

## Shape Classification and Segmentation

Employing 3DCNN [3] to classify shapes on ShapeNet dataset. The model uses a number of MLP layers and finally gets a prediction by partial object and by whole object.

![alt text](E2/exercise_2/images/3dcnn.png)

PointNet [4] is used to for classification of 3D point clouds. The model could also be used for semantic segmentation by aggregating global features with the intermediate representation.

![alt text](E2/exercise_2/images/pointnet.png)

## Surface Representation and Alignment

Converting between different 3D surface representations, aligning shapes using Procrustes, and saving visualizations.

### Signed Distance Fields

Given a point calculate the SDF function for

- sphere
- torus
- atom: combines sphere and torus SDF function
- SDF grids: build an SDF grid that shows the implicit representation in a cube of size 1x1x1
- Visualization of `.ply` objects in `MeshLab`

### Conversion from SDF Grids to Occupancy Grids

convert implicit SDF representation to explicit occupancy grid representation, where each voxel is either occupied or not. This is done by thresholding the SDF values on zero.

### Conversion From SDF to Triangle Meshes

- Implementation of `Marching Cubes` where it iterates over all cubes in a grid:
  - compute 8-bit index representing corners of the cube
  - retrieve triangle intersections from a precomputed table using the index
  - add vertices and faces to global list
- Implementation of vertex interpolation to get a smoother representation of the surface.
- Write triangle mesh to `Wavefront OBJ` file

Sphere            |  Torus
:-------------------------:|:-------------------------:
![sphere](./E1/images/marching-cubes-sphere.png) | ![torus](./E1/images/marching-cubes-torus.png)

Atom            |  MLP
:-------------------------:|:-------------------------:
![atom](./E1/images/marching-cubes-atom.png) | ![mlp](./E1/images/marching-cubes-mlp.png)

### Conversion from Triangle Meshes to Point Clouds

- Convert triangle mesh to point cloud by sampling n points using barycentric coordinates for each triangle
- Export point cloud as `Wavefront OBJ`

![sphere](./E1/images/atom-point-cloud.png)

### Rigid Shape Alignment with Procrustes

Align two shapes given near-perfect point correspondences using procrustes

- Center both shapes
- estimate rotation
- estimate translation

Before            |  After
:-------------------------:|:-------------------------:
![before](./E1/images/alignment-before.png) | ![after](./E1/images/alignment-after.png)

## 3D Shape Reconstruction

Using 3D-Encoder-Predictor CNNs (3D-EPN) [1] which takes incomplete shape observation as distance field and state and encodes it into a latent space then uses a predictor to generate unsigned distance field of the complete shape. The output shape is further enhanced using a post-processing step where a shape database is used to sample parts to increase the output shape resolution.

![alt text](E3/exercise_3/images/3depn_teaser.png)

## References

[1] Dai, Angela, Charles Ruizhongtai Qi, and Matthias Nießner. "Shape completion using 3d-encoder-predictor cnns and shape synthesis." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

[2] Park, Jeong Joon, et al. "Deepsdf: Learning continuous signed distance functions for shape representation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[3] Qi, C. et al. “Volumetric and Multi-view CNNs for Object Classification on 3D Data.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 5648-5656.

[4] Qi, C. et al. “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 77-85.

[5] Park, Jeong Joon, et al. "Deepsdf: Learning continuous signed distance functions for shape representation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019

[6] Mescheder, Lars, et al. "Occupancy networks: Learning 3d reconstruction in function space." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[7] Lorensen, William E., and Harvey E. Cline. "Marching cubes: A high resolution 3D surface construction algorithm." ACM siggraph computer graphics 21.4 (1987): 163-169.

[8] Schönemann, Peter H. "A generalized solution of the orthogonal procrustes problem." Psychometrika 31.1 (1966): 1-10.
