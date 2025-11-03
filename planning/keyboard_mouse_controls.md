# Keyboard & Mouse Controls Guide

**Last Updated:** 2025-10-22
**Purpose:** Desktop/Editor control for testing and alternative teacher input
**Script:** KeyboardMouseTerrainController.cs

---

## 🖱️ Mouse Controls

### Overview:

| Action | Control | Visual | Description |
|--------|---------|--------|-------------|
| **Pan** | Left Click + Drag | Green ray | Slide terrain across surface |
| **Rotate** | Right Click + Drag | Yellow ray | Spin around cursor location |
| **Zoom** | Mouse Wheel | Blue ray | Scroll up=zoom in, down=zoom out |
| **Annotate** | Middle Click | Red ray | Place marker at cursor |

---

### Detailed Instructions:

#### 1. Pan Terrain (Green Ray)

**Control:** Left Click + Drag

**How to Use:**
1. Hover mouse over terrain
2. **Hold Left Mouse Button**
3. **Drag mouse** in any direction
4. Terrain slides following mouse movement
5. Release to stop panning

**Behavior:**
- Movement projected onto table surface (horizontal plane)
- Constrained to 5 units from anchor point
- Smooth, responsive control

---

#### 2. Rotate Terrain (Yellow Ray)

**Control:** Right Click + Drag

**How to Use:**
1. Hover mouse over terrain (marks rotation center)
2. **Hold Right Mouse Button**
3. **Drag mouse left/right**
   - Drag **right** → Rotate clockwise
   - Drag **left** → Rotate counter-clockwise
4. Release to stop rotating

**Behavior:**
- Rotation axis passes through cursor position on terrain
- Vertical axis only (perpendicular to table)
- Speed based on horizontal mouse movement

---

#### 3. Zoom Terrain (Blue Ray)

**Control:** Mouse Wheel Scroll

**How to Use:**
1. Hover mouse over terrain (marks zoom center)
2. **Scroll wheel UP** → Zoom IN (terrain gets bigger)
3. **Scroll wheel DOWN** → Zoom OUT (terrain gets smaller)

**Behavior:**
- Zoom centered on cursor position (like Google Maps!)
- Terrain also moves slightly toward cursor while zooming
- Limits: 0.5x (min) to 3.0x (max) scale

---

#### 4. Place Marker (Red Ray)

**Control:** Middle Mouse Click

**How to Use:**
1. Hover mouse over desired location on terrain
2. **Click Middle Mouse Button** (wheel click)
3. Red marker pin appears at cursor location
4. Auto-numbered: "Pin 1", "Pin 2", etc.

**Behavior:**
- Instant placement
- Maximum 50 markers
- All users see the marker

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| **Ctrl + Z** | Undo Marker | Remove last placed marker |
| **R** | Reset Terrain | Return to default position/rotation/scale |
| **Esc** | Stop All | Cancel any ongoing manipulation |
| **1, 2, 3, 4** | Debug Modes | Show mode in console (for debugging) |

---

## 🎯 Use Cases

### 1. Unity Editor Testing

**Fastest way to test without VR:**

1. Open Unity scene in Play mode
2. Use mouse to manipulate terrain
3. Test network synchronization
4. Debug marker placement
5. Verify all features work

**Benefits:**
- No need to build to Quest every time
- Instant iteration
- Easy debugging with Console logs
- Can use Unity Inspector while testing

---

### 2. Desktop Teacher Mode

**Teacher controls from PC, students in VR:**

**Setup:**
- Teacher runs on desktop (no headset)
- Students wear Quest headsets
- All connected to same network
- Teacher uses keyboard/mouse, students view in VR

**Use Case:**
- Remote teaching
- Teacher prefers desktop
- Large display presentation
- Screen recording/streaming

---

### 3. Hybrid Classroom

**Mix of VR and desktop participants:**

- Some students in VR (immersive)
- Some students watching desktop screen (projected)
- Teacher alternates between VR and desktop
- All see same terrain in real-time

---

### 4. Development & Debugging

**Quick testing workflow:**

```
1. Edit script in Visual Studio/Rider
2. Switch to Unity
3. Press Play
4. Use mouse to test changes
5. See results immediately
6. Iterate quickly
```

**Much faster than:** Build → Deploy to Quest → Test → Repeat

---

## 🔄 Automatic Input Detection

The system can auto-detect input method:

```csharp
// Pseudo-code for auto-detection
if (XRDevice.isPresent && VRHeadsetActive)
{
    // Use VR controller/hand tracking
    KeyboardMouseController.SetEnabled(false);
}
else
{
    // Use keyboard and mouse
    KeyboardMouseController.SetEnabled(true);
}
```

**Result:** Seamless switching between VR and desktop control!

---

## 🎨 Visual Feedback

### Mouse Ray Colors:

- **Cyan:** Idle (cursor over terrain)
- **Green:** Left dragging (panning)
- **Yellow:** Right dragging (rotating)
- **Blue:** Scrolling wheel (zooming)
- **Red:** Middle clicking (annotating)

### Cursor Hit Indicator:

- Small disc at cursor position on terrain
- Shows exactly where actions will be centered
- Visible only when hovering over terrain

---

## 🔧 Unity Setup

### Option 1: Editor Testing (Recommended)

**Add to Terrain GameObject:**

1. Select Terrain in Hierarchy
2. Add Component → KeyboardMouseTerrainController
3. Assign references:
   - Terrain Manager: TerrainInteractionManager component
   - Teacher Control: (leave empty for testing)
   - Terrain Transform: Terrain itself
   - Annotation System: AnnotationSystem in scene
   - Main Camera: Main Camera or OVRCameraRig/CenterEyeAnchor

4. Configure:
   - Enable Keyboard Mouse: ☑
   - Show Mouse Ray: ☑

5. Press Play and test!

---

### Option 2: Desktop Teacher Build

**Add to Player Prefab:**

1. Open Player prefab
2. Add Component → KeyboardMouseTerrainController
3. Assign all references
4. Add auto-detection script:

```csharp
void Start()
{
    bool isVR = XRSettings.isDeviceActive;
    GetComponent<KeyboardMouseTerrainController>().SetEnabled(!isVR);
    GetComponent<PointerBasedTerrainController>().enabled = isVR;
}
```

5. Build for Windows (in addition to Android build)

---

## 🎮 Control Comparison

### VR Controller vs Keyboard/Mouse:

| Feature | VR Controller | Keyboard/Mouse |
|---------|--------------|----------------|
| **Zoom** | Trigger + Push/Pull | Mouse Wheel |
| **Pan** | Grip + Move hand | Left Click + Drag |
| **Rotate** | Trigger + Thumbstick | Right Click + Drag |
| **Annotate** | Button A | Middle Click |
| **Undo** | Button B | Ctrl + Z |
| **Precision** | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Very High |
| **Speed** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Very Fast |
| **Immersion** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Limited |
| **Fatigue** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Very Low |

**When to use each:**

**VR Controller:**
- Classroom teaching (immersive)
- Student demonstrations
- Physical presence required

**Keyboard/Mouse:**
- Unity Editor testing (fastest!)
- Remote teaching
- Desktop presentations
- Screen recordings
- Long sessions (less fatigue)

---

## 🧪 Testing Workflow

### Quick Editor Testing (No VR):

1. **Open Unity scene**
2. **Press Play** (▶)
3. **Game view becomes active**
4. **Test controls:**
   - Left drag → Pan works?
   - Right drag → Rotate works?
   - Scroll wheel → Zoom works?
   - Middle click → Marker appears?
   - Ctrl+Z → Undo works?

5. **Check Console** for debug messages
6. **Press Play again** to stop

**Time:** 5 seconds to test vs 5+ minutes to build to Quest!

---

### Multi-User Testing (Desktop + VR):

**Setup:**
1. Build VR version to Quest (student client)
2. Run Unity Editor as Host (desktop teacher)
3. Quest joins as Client

**Test:**
1. Desktop: Click "Start as Host"
2. Quest: Join as Client
3. Desktop: Use mouse to manipulate terrain
4. **VR student sees changes in real-time!**
5. Verify network synchronization

---

## 💡 Pro Tips

### Smooth Zoom-to-Cursor:

The script includes "zoom to cursor" behavior:
- As you zoom in, terrain moves toward cursor
- Feels like Google Maps or CAD software
- Natural for desktop users

### Quick Marker Placement:

For rapid annotation:
1. Hover → Middle click
2. Move → Middle click
3. Move → Middle click
4. Place multiple markers quickly!

### Precise Rotation:

For exact angles:
1. Right click + drag slowly
2. Use small mouse movements
3. More precise than VR thumbstick

### Testing Network Features:

Use keyboard/mouse in editor to test:
- Teacher role assignment
- Marker synchronization
- Authority management
- Student view-only mode

---

## 🔄 Advanced: Input Method Switching

### Auto-Detection Script:

```csharp
using UnityEngine;
using UnityEngine.XR;

public class InputMethodDetector : MonoBehaviour
{
    private KeyboardMouseTerrainController kbmController;
    private PointerBasedTerrainController vrController;

    void Start()
    {
        kbmController = GetComponent<KeyboardMouseTerrainController>();
        vrController = GetComponent<PointerBasedTerrainController>();

        UpdateInputMethod();
    }

    void Update()
    {
        // Check every second
        if (Time.frameCount % 60 == 0)
        {
            UpdateInputMethod();
        }
    }

    void UpdateInputMethod()
    {
        bool isVR = XRSettings.isDeviceActive;

        if (kbmController != null)
            kbmController.SetEnabled(!isVR);

        if (vrController != null)
            vrController.enabled = isVR;

        Debug.Log($"Input method: {(isVR ? "VR" : "Desktop")}");
    }
}
```

**Add this to Player prefab for automatic switching!**

---

## 📋 Setup Checklist

**For Editor Testing:**
- [ ] KeyboardMouseTerrainController added to Terrain
- [ ] All references assigned
- [ ] Enable Keyboard Mouse: ☑
- [ ] Show Mouse Ray: ☑
- [ ] Press Play and test

**For Desktop Teacher Mode:**
- [ ] KeyboardMouseTerrainController added to Player prefab
- [ ] Build for Windows (File → Build Settings → PC, Mac & Linux)
- [ ] Create desktop executable
- [ ] Test with VR clients

**For Hybrid Setup:**
- [ ] Add InputMethodDetector script
- [ ] Both controllers on Player prefab
- [ ] Auto-detection enabled
- [ ] Tested in both VR and Desktop modes

---

## 🎓 Comparison: All Input Methods

| Feature | Quest Controller | Hand Gestures | Keyboard/Mouse |
|---------|-----------------|---------------|----------------|
| **Setup** | VR headset | VR headset | Desktop PC |
| **Precision** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Immersion** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Fatigue** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Learning** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Testing** | ⭐⭐ Slow | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Fast |

**Recommendation:**
- **Development/Testing:** Keyboard/Mouse (fastest iteration)
- **Teaching (In-Person):** VR Controller or Hand Gestures
- **Remote Teaching:** Keyboard/Mouse from desktop
- **Best:** Support all three! (already implemented)

---

## 🚀 Quick Start

**To test right now in Unity Editor:**

1. Open Unity scene
2. Select Terrain GameObject
3. Add Component → KeyboardMouseTerrainController
4. Drag Terrain to "Terrain Manager" field
5. Drag Terrain to "Terrain Transform" field
6. Find AnnotationSystem in scene, drag to "Annotation System" field
7. Find Main Camera, drag to "Main Camera" field
8. Press Play ▶
9. Use mouse to control terrain!

**No VR headset needed for testing!**

---

**End of Keyboard & Mouse Controls Guide**