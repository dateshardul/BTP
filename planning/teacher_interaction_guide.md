# Teacher Interaction Guide - IMPROVED Controls

**Last Updated:** 2025-10-22
**Device:** Meta Quest 3 with Physical Controllers OR Hand Tracking
**User Roles:** Teacher (control) vs Student (view only)
**Control Scheme:** IMPROVED for better UX

---

## 🎮 IMPROVED Controller Layout (Meta Quest 3)

### Right Controller Controls:

```
┌─────────────────────────────────────┐
│     Meta Quest 3 Right Controller   │
├─────────────────────────────────────┤
│                                     │
│  [B Button]  ←  Undo Last Marker    │
│  [A Button]  ←  Place Marker (RED)  │
│                                     │
│  [Thumbstick]                       │
│    - With Trigger: Rotate ← → (YEL)│
│                                     │
│  [Index Trigger]                    │
│    - Hold + Move: Zoom (BLUE)       │
│    - Hold + Stick: Rotate (YELLOW)  │
│                                     │
│  [Grip/Middle]                      │
│    - Hold + Move: Pan (GREEN)       │
│                                     │
└─────────────────────────────────────┘
```

### Control Summary:

| Action | Control | Pointer Color | Description |
|--------|---------|---------------|-------------|
| **Place Marker** | Button A (single press) | Red | Instantly place pin at pointer |
| **Undo Marker** | Button B (single press) | Red | Remove last placed marker |
| **Pan Terrain** | Grip + Move controller | Green | Slide terrain on surface |
| **Zoom Terrain** | Trigger + Move controller | Blue | Closer=zoom in, farther=zoom out |
| **Rotate Terrain** | Trigger + Thumbstick ←→ | Yellow | Spin around pointer point |

---

## 🎯 How Each Control Works

### 1. Zoom (Blue Pointer) - **Trigger + Move**

**Purpose:** Scale the terrain larger or smaller

**How to Use:**
1. Point controller at desired zoom center on terrain
2. **Hold Index Trigger**
3. **Move controller** toward/away from terrain:
   - **Move CLOSER** → Zoom IN (terrain gets bigger)
   - **Move FARTHER** → Zoom OUT (terrain gets smaller)
4. Release trigger to stop zooming

**Key Features:**
- Zoom centers on pointer hit point (not terrain center!)
- Natural push-pull motion
- Limits: 0.5x (min) to 3.0x (max) scale

**UX Benefit:** Feels like physically pushing/pulling the terrain

---

### 2. Pan (Green Pointer) - **Grip + Move**

**Purpose:** Slide the terrain across the table surface

**How to Use:**
1. Point controller at terrain
2. **Hold Grip (middle trigger)**
3. **Move controller** left/right/forward/back
4. Terrain follows your hand movement
5. Release grip to stop panning

**Key Features:**
- Movement constrained to surface plane (stays on table)
- 1:1 controller-to-terrain movement mapping
- Limit: 5 units from anchor point

**UX Benefit:** Feels like physically dragging the map across a table

---

### 3. Rotate (Yellow Pointer) - **Trigger + Thumbstick**

**Purpose:** Spin the terrain around a vertical axis

**How to Use:**
1. Point controller at desired rotation center
2. **Hold Index Trigger**
3. **Push Thumbstick LEFT** or **RIGHT**:
   - **LEFT** (←) → Rotate counter-clockwise
   - **RIGHT** (→) → Rotate clockwise
4. Release trigger or center thumbstick to stop

**Key Features:**
- Rotation axis passes through pointer hit point
- Vertical axis only (perpendicular to table)
- Speed controlled by how far you push thumbstick

**UX Benefit:** Easy to control rotation speed, no need to twist wrist

---

### 4. Annotate (Red Pointer) - **Button A**

**Purpose:** Place red marker pins on the terrain

**How to Use:**
1. Point controller at desired marker location on terrain
2. **Press Button A** once
3. Red marker pin appears instantly
4. Marker has auto-numbered label: "Pin 1", "Pin 2", etc.

**Undo:**
- **Press Button B** to remove the last placed marker
- Can undo multiple times

**Key Features:**
- Instant placement (no hold required)
- Haptic feedback when placed
- Maximum 50 markers per session
- All markers visible to all users

**UX Benefit:** Quick one-button placement, easy undo

---

## 👋 Hand Gesture Controls (Alternative to Controller)

**For teachers who prefer hands-free interaction**

### Gesture Overview:

| Gesture | Action | Description |
|---------|--------|-------------|
| **Two-Hand Pinch + Spread** | Zoom | Pinch both hands, spread apart = zoom in, bring together = zoom out |
| **Single Hand Grab** | Pan | Pinch with one hand, move hand = terrain follows |
| **Point + Dwell** | Annotate | Point index finger at location for 1 second = place marker |
| **Two-Hand Twist** | Rotate | Pinch both hands, twist = terrain rotates |

---

### 1. Pinch-Zoom Gesture (Two Hands)

**How to Use:**
1. Hold both hands up, palms facing each other
2. **Pinch** index finger to thumb on **both hands**
3. **Spread hands apart** → Zoom IN
4. **Bring hands together** → Zoom OUT
5. Release pinch to stop

**Visual Feedback:** Green rays from both index fingers

---

### 2. Grab-Pan Gesture (One Hand)

**How to Use:**
1. **Pinch** index finger to thumb (one hand)
2. **Move your hand** left/right/forward/back
3. Terrain follows your hand movement
4. Release pinch to stop

**Visual Feedback:** Green ray from index finger

---

### 3. Point-Annotate Gesture (One Hand)

**How to Use:**
1. **Point** index finger at location on terrain
2. **Hold steady** for 1 second (dwell time)
3. Marker automatically places
4. Preview sphere shows where marker will appear

**Visual Feedback:**
- Red ray from index finger
- Growing sphere at target (indicates dwell progress)
- Haptic pulse when placed

---

### 4. Twist-Rotate Gesture (Two Hands)

**How to Use:**
1. **Pinch both hands** (index to thumb)
2. **Twist hands** clockwise or counter-clockwise
3. Terrain rotates around center point between hands
4. Release to stop

**Visual Feedback:** Yellow rays from both hands

---

## 🔄 Switching Between Controller and Hand Tracking

The system automatically switches:

- **Controller Detected:** Controller controls active, hand gestures disabled
- **Hands Only:** Hand gesture controls active
- **Both Available:** Controller takes priority (can switch via menu)

**Advantage:**
- Teachers can put down controller and use hands naturally
- Pick up controller for more precise control
- Seamless switching mid-session

---

## 🎨 Visual Feedback System

### Pointer Ray Colors:

**Controller Mode:**
- **Cyan:** Idle (not pressing anything)
- **Blue:** Index Trigger pressed (Zoom mode)
- **Green:** Grip pressed (Pan mode)
- **Yellow:** Trigger + Thumbstick (Rotate mode)
- **Red:** Button A pressed (Annotate mode)

**Hand Gesture Mode:**
- **Cyan:** Hands visible but no gesture
- **Green:** Single pinch (Grab-pan)
- **Blue:** Two-hand pinch spread apart (Zoom)
- **Yellow:** Two-hand pinch rotating (Rotate)
- **Red:** Pointing steady (Annotate dwell)

### Hit Indicator:
- Circular disc where pointer hits terrain
- Pulses when active
- Aligns with surface normal
- Larger when annotating

---

## 🎓 Controller vs Hand Gestures Comparison

| Feature | Controller | Hand Gestures |
|---------|-----------|---------------|
| **Precision** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐ Medium |
| **Speed** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Fast |
| **Natural Feel** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Fatigue** | ⭐⭐⭐⭐ Low | ⭐⭐⭐ Medium (arms up) |
| **Learning Curve** | ⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Intuitive |
| **Reliability** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good (lighting dependent) |

**Recommendation:**
- **Controller:** For precise annotations, detailed work
- **Hand Gestures:** For demonstrations, natural teaching flow
- **Best:** Switch between both as needed!

---

## 📋 Role-Based Access Control

### Teacher Role:
- Can use controller OR hand gestures
- All 4 controls available (Zoom, Pan, Rotate, Annotate)
- Pointer ray visible
- Can place/remove markers
- Teacher control UI displayed

### Student Role:
- **View only** - cannot manipulate terrain
- No pointer ray
- No control UI
- See teacher's actions in real-time
- See all markers

### Role Assignment:
- Default: All users start as Students
- Host/Server promotes to Teacher role
- Typically: 1 teacher, multiple students

---

## 🏗️ Surface Anchoring

### Automatic Detection:
- Uses Meta Quest's Scene Understanding
- Detects tables, desks, floors automatically
- Anchors terrain to nearest horizontal surface
- Terrain hovers 5cm above surface

### Supported Surfaces:
- Tables / Desks / Countertops
- Floor (fallback)
- Any flat horizontal surface (>80° from vertical)

### Behavior:
- All panning constrained to surface plane
- Rotation always around vertical axis
- Maintains position relative to physical table

---

## 📍 Annotation System

### Marker Features:
- **Google Maps style:** Red sphere top, vertical stick
- **Auto-numbered:** "Pin 1", "Pin 2", etc.
- **Persistent:** Stays until manually removed
- **Networked:** All users see same markers
- **Billboard labels:** Always face camera
- **Maximum:** 50 markers per session

### Marker Management:
| Action | Control | Description |
|--------|---------|-------------|
| Place | Button A | Instant placement at pointer |
| Undo | Button B | Remove last marker |
| Clear All | UI Button | Remove all markers |

---

## Troubleshooting

### Pointer Not Showing:
- Check if user is Teacher role
- Verify LineRenderer component assigned
- Check controller connection

### Can't Manipulate Terrain:
- Confirm user has Teacher role
- Check TerrainInteractionManager reference
- Verify NetworkObject ownership

### Markers Not Appearing:
- Check MarkerPin prefab assigned
- Verify prefab in Network Prefabs List
- Check if marker limit (50) reached

### Terrain Not Anchored:
- Wait for room scanning to complete
- Check MR Utility Kit setup
- Verify surface detection layer

### Wrong Rotation Center:
- Make sure pointer is hitting terrain
- Check hit indicator position
- Verify ray is hitting correct layer

---

## Advanced Features (Future)

- Custom marker labels (text input)
- Marker colors for categories
- Marker groups/layers
- Drawing lines between markers
- Voice annotations
- Screenshot/save session
- Marker templates

---

## Keyboard Shortcuts (Testing in Editor)

For testing without VR controller:

- **1 Key:** Zoom mode
- **2 Key:** Pan mode
- **3 Key:** Rotate mode
- **4 Key:** Annotate mode
- **Mouse:** Simulate pointer
- **Left Click:** Trigger
- **WASD:** Move controller position
- **Q/E:** Rotate controller

---

**End of Teacher Interaction Guide**