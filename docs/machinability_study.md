# 3-Axis & 5-Axis CNC Machinability Intelligence & Additive Manufacturing Constraints
------------------------------------------------------------------------------------
**Author**: WNFEA Core Engineering Team  
**Date**: September 2026  
**Status**: Technical Architecture Specification & Roadmap  
**Target Milestone**: WNFEA Manufacturing Intelligence Engine  

---

## 1. Executive Summary

A pervasive failure of standard topology optimization and generative design algorithms is the generation of organic, unmachinable geometries. While optimal from a theoretical strain energy perspective, unconstrained generative results typically feature:
1. **Severe Internal Cavities & Undercuts**: Geometries that cannot be accessed by rotating milling cutters or EDM wires.
2. **Sub-Millimeter Internal Radii**: Fillets sharper than standard endmill radii ($r < R_{cutter}$), requiring micro-tooling or expensive manual benching.
3. **Infinite-Axis Requirements**: Shapes that demand continuous 5-axis simultaneous simultaneous toolpaths with severe gouging risks and exorbitant machine-hour costs.

### The WNFEA Manufacturing-First Framework
WNFEA embeds **continuous, differentiable manufacturing constraints** directly into the finite element optimization and adaptive refinement loops. Rather than treating machinability as an afterthought (post-processing filtering or manual CAD reconstruction), WNFEA formulates:
* **Parametric Tool Visibility Cones & Line-of-Sight Operators** for 3-axis, 3+2 axis, and continuous 5-axis milling.
* **Morphological Opening Operators** guaranteeing minimum feature size and endmill radius conformance.
* **Automated Parametric Feature Synthesis (Fusion 360 Style)** translating continuous density fields into editable FreeCAD sketches, pockets, bosses, and standardized fillet radii.

```
+---------------------------------------------------------------------------+
|                   WNFEA MANUFACTURING CONSTRAINED PIPELINE                |
|                                                                           |
|   +---------------------+           +------------------------+            |
|   |  Unfiltered Density | --------> |  CNC Visibility Filter |            |
|   |  rho_e in [0, 1]    |           |  (Ray-Casting Cone Op) |            |
|   +---------------------+           +------------------------+            |
|                                                  |                        |
|                                                  v                        |
|   +---------------------+           +------------------------+            |
|   |  Machinable Density | <-------- |  Morphological Opening |            |
|   |  rho_mach in [0, 1] |           |  (r >= R_endmill)      |            |
|   +---------------------+           +------------------------+            |
|              |                                                            |
|              v                                                            |
|   +----------------------------------------------------------+            |
|   |           AUTOMATED PARAMETRIC CAD FEATURE SYNTHESIS     |            |
|   |  Planar Pocket Extraction --> Standard Endmill Sizing    |            |
|   |  --> FreeCAD Sketcher / PartDesign Parametric Solid      |            |
+---+----------------------------------------------------------+------------+
```

---

## 2. Mathematical Formulation of CNC Milling Constraints

### 2.1 Tool Clearance & Line-of-Sight Operator
Let the cutting tool be modeled by a rigid cylindrical spindle and shank along unit approach vector $\mathbf{v} \in \mathbb{S}^2$ with cutter radius $R_c$ and cutting length $L_c$.

A point $\mathbf{x} \in \Omega$ is machinable from direction $\mathbf{v}$ if and only if the semi-infinite cylinder $\mathcal{C}(\mathbf{x}, \mathbf{v}, R_c)$ swept outward along $\mathbf{v}$ contains no workpiece material:
$$\mathcal{C}(\mathbf{x}, \mathbf{v}, R_c) = \left\{ \mathbf{y} \in \mathbb{R}^3 \;\middle|\; (\mathbf{y} - \mathbf{x}) \cdot \mathbf{v} > 0, \; \|(\mathbf{y} - \mathbf{x}) - ((\mathbf{y} - \mathbf{x}) \cdot \mathbf{v})\mathbf{v}\| \le R_c \right\}$$

#### Differentiable Milling Density Projection
For 3-axis milling along spindle axis $\mathbf{v} = +z$ (top face) and $-\mathbf{v} = -z$ (bottom face), the solid density must be monotonically non-decreasing or non-increasing along the tool path to avoid internal undercuts:
$$\frac{\partial \rho}{\partial z} \le 0 \quad (\text{Top Milling}) \quad \text{or} \quad \frac{\partial \rho}{\partial z} \ge 0 \quad (\text{Bottom Milling})$$

In discretized continuum mechanics, this is enforced by a continuous forward/backward propagation operator:
$$\rho_{mach, e}^{(+z)} = \max \left( \rho_e, \rho_{above, e} \right)$$
Using a differentiable smooth approximation (Soft-Maximum / LogSumExp):
$$\rho_{mach}(\mathbf{x}) = \frac{1}{\alpha} \ln \left( \int_{z}^{z_{max}} \exp(\alpha \, \rho(x, y, \zeta)) \, d\zeta + \exp(\alpha \, \rho_{base}) \right)$$
where $\alpha \gg 1$ is a sharpness parameter. This formulation provides exact analytic gradients $\frac{\partial \rho_{mach}}{\partial \rho_e}$ for adjoint sensitivity backpropagation.

---

### 2.2 Endmill Radius Conformance via Morphological Filtering
A physical CNC milling cutter cannot machine internal corners sharper than its own radius $R_{tool}$. Sharp concave corners generate severe stress concentrations and cannot be physically manufactured.

In WNFEA, minimum radius conformance is enforced mathematically using morphological mathematical morphology operations over the density field:

#### 1. Continuous Erosion ($\ominus$)
$$\rho_{eroded}(\mathbf{x}) = \min_{\mathbf{y} \in \mathcal{B}(\mathbf{x}, R_{tool})} \rho(\mathbf{y}) \approx - \frac{1}{\alpha} \ln \left( \int_{\mathcal{B}(\mathbf{x}, R_{tool})} \exp(-\alpha \, \rho(\mathbf{y})) \, d\mathbf{y} \right)$$

#### 2. Continuous Dilation ($\oplus$)
$$\rho_{opened}(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{B}(\mathbf{x}, R_{tool})} \rho_{eroded}(\mathbf{y}) \approx \frac{1}{\alpha} \ln \left( \int_{\mathcal{B}(\mathbf{x}, R_{tool})} \exp(\alpha \, \rho_{eroded}(\mathbf{y})) \, d\mathbf{y} \right)$$

The **Morphological Opening**:
$$\boldsymbol{\rho}_{mach} = (\boldsymbol{\rho} \ominus \mathcal{B}_{R_{tool}}) \oplus \mathcal{B}_{R_{tool}}$$
guarantees that:
1. Every void pocket can fit an endmill of radius $R_{tool}$.
2. Every internal concave fillet has a radius $r \ge R_{tool}$.
3. Narrow thin ribs narrower than the cutter kerf are automatically suppressed.

---

## 3. 5-Axis Continuous & 3+2 Indexed Machining Formulations

In continuous 5-axis machining, the tool vector $\mathbf{v}(\theta, \phi)$ can dynamically reorient within the machine tool's physical rotary kinematic table/head limits:
$$A_{min} \le \theta \le A_{max}, \quad C_{min} \le \phi \le C_{max}$$

### 3.1 Parametric Visibility Cones
For each surface element $e$, WNFEA constructs the spherical accessibility hemisphere $\mathcal{S}_e \subset \mathbb{S}^2$:
$$\mathcal{S}_e = \left\{ \mathbf{v} \in \mathbb{S}^2 \;\middle|\; \mathbf{v} \cdot \mathbf{n}_e \ge \cos(\psi_{max}), \; \text{Ray}(\mathbf{x}_e, \mathbf{v}) \cap \Omega_{solid} = \emptyset \right\}$$
where:
* $\mathbf{n}_e$ is the outward surface normal.
* $\psi_{max}$ is the maximum allowable tool lead/lag angle before spindle housing interference.

### 3.2 3+2 Indexed Setup Optimization
Rather than full continuous 5-axis (which increases CNC programming complexity and machine hourly rates by $3\times$ to $5\times$), industrial manufacturing prefers **3+2 indexed machining** (machining from a discrete set of 3 to 6 fixed orientations).

WNFEA formulates indexed orientation selection as a mixed-integer facility location optimization:
$$\min_{\{\mathbf{v}_1, \dots, \mathbf{v}_K\}} \quad \sum_{e \in \partial \Omega} \min_{k \in \{1 \dots K\}} \left( 1 - \mathcal{A}(\mathbf{x}_e, \mathbf{v}_k) \right)$$
* Automatically identifies the optimal stock fixturing coordinate systems (G54, G55, G56).
* Minimizes unmachinable residual material while restricting the setup count to $K \le 4$.

---

## 4. Additive Manufacturing (AM / 3D Printing) Constraints

For components targeted for Selective Laser Melting (SLM) or Fused Deposition Modeling (FDM):

### 4.1 Overhang Self-Support Angle Constraint
Material deposited without underlying support will collapse or warp if the overhang angle with the horizontal build plate exceeds critical angle $\theta_c$ (typically $45^\circ$ for metal SLM):
$$\mathbf{n}(\mathbf{x}) \cdot \mathbf{v}_{build} \ge - \cos(\theta_c)$$
WNFEA expresses this as a continuous local slope penalty:
$$\mathcal{P}_{overhang}(\boldsymbol{\rho}) = \int_{\Omega} \left( \max\left( 0, -\nabla \rho \cdot \mathbf{v}_{build} - \|\nabla \rho\| \cos(\theta_c) \right) \right)^2 d\Omega$$
Minimizing this penalty eliminates the need for sacrificial internal support structures, cutting post-processing EDM and machining costs to zero.

---

## 5. Direct Parametric CAD Reconstruction (Fusion 360 Style)

A major innovation of the WNFEA manufacturing pipeline is translating optimized density fields into **native, fully editable FreeCAD parametric features**:

```
Optimized Machinable Density Field
               |
               v
Planar Face Detection (RANSAC Normal Clustering)
  - Identifies top/bottom stock datum planes
  - Extracts outer profile and internal pocket boundaries
               |
               v
2D Boundary Polygonal Approximation & Curve Fitting
  - Replaces jagged pixel contours with tangent-continuous lines and circular arcs
  - Enforces r >= R_tool on all internal fillets
               |
               v
FreeCAD 1.1 PartDesign Synthesis:
  - PartDesign::Body
  - Base Pad: Extrude outer boundary from stock billet
  - Pocket Features: Extrude cut internal pockets along tool axis
  - Standard Fillets: Apply Fillet feature with R = R_tool
               |
               v
Fully Parametric Editable STEP / FreeCAD (.FCStd) File
```

### Advantages Over Mesh / STL Output
1. **Direct Toolpath Generation**: CAM engines (Mastercam, Fusion 360 CAM, FreeCAD CAM) can directly select planar faces and analytical cylindrical boundaries to generate 2.5D roughing and finishing paths in seconds.
2. **Design Intent Preservation**: Engineers can open the resulting `.FCStd` file, adjust a pocket depth or bolt hole diameter, and the parametric tree regenerates cleanly without boolean mesh corruption.
3. **Zero Porosity**: 100% solid B-Rep volumes with mathematically verified watertightness.

---

## 6. Verification Plan & Industrial Case Study

To validate the manufacturing intelligence engine:
1. **Benchmark Component**: Aerospace engine gimbal bracket ($300\text{ mm} \times 150\text{ mm} \times 80\text{ mm}$, 7075-T6 Aluminum).
2. **Loads**: $15\text{ kN}$ bearing load + $25g$ lateral acceleration field.
3. **Tool Constraints**: 3-axis CNC vertical mill, $12\text{ mm}$ flat endmill ($R_{tool} = 6\text{ mm}$), maximum pocket depth $50\text{ mm}$.
4. **Validation**:
   * Verify all internal fillets satisfy $r \ge 6.0\text{ mm}$.
   * Verify zero undercut faces ($\frac{\partial \rho}{\partial z} \le 0$).
   * Post-process into FreeCAD CAM workbench and generate G-code verifying zero tool shank collisions.
