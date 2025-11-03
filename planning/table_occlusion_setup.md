# Table Edge Occlusion Setup Guide

**Last Updated:** 2025-10-22
**Feature:** Hide terrain parts that extend beyond physical table edges
**Methods:** Automatic (Recommended) + Manual Backup

---

## 🎯 The Problem

When teacher zooms in or rotates the terrain, parts of the virtual map may extend beyond the physical table edges. This looks unrealistic in Mixed Reality.

**What we want:**
- ✅ Terrain parts ON the table → Visible
- ❌ Terrain parts BEYOND table edges → Hidden/Clipped

---

## ✨ Solution 1: Meta Quest 3 Automatic Occlusion (RECOMMENDED)

### What Is It?

Meta Quest 3 has **built-in depth cameras** that:
1. Scan the physical environment in real-time
2. Detect all surfaces (tables, walls, floors, etc.)
3. Automatically hide virtual objects behind real objects
4. **This happens automatically - no coding needed!**

### How It Works:

```
Meta Quest 3 Cameras
    ↓ [Depth scanning]
Detects table edges
    ↓ [Real-time occlusion]
Virtual terrain beyond table = Hidden
Virtual terrain on table = Visible
```

**Result:** Parts of terrain beyond table edges are **automatically clipped** in real-time!

---

## 🔧 Setup Instructions (Unity)

### Step 1: Enable Scene Support

**When starting app on Quest:**

1. **First Time Only:** User must complete room setup
   - Put on Quest headset
   - App will prompt: "Set up your space"
   - Follow on-screen instructions
   - Quest scans the room and detects surfaces
   - This takes 1-2 minutes

2. **App Permissions:**
   - App will request "Scene" permission
   - User must Allow
   - This lets app access detected surfaces

---

### Step 2: Configure OVRManager in Unity

**This enables depth-based occlusion**

1. **Find OVRManager:**
   - In Hierarchy, expand "OVRCameraRig"
   - Click on the root "OVRCameraRig" object
   - In Inspector, find **OVRManager** component (should be there by default)

2. **Enable Depth Submission:**
   - In OVRManager, find "Quest Features" section
   - Check **☑ Depth Submission** (CRITICAL!)
   - This sends depth data for occlusion

3. **Enable Scene Support:**
   - Still in OVRManager Quest Features
   - Check **☑ Scene Support**
   - This enables Scene API (table detection)

4. **Optional - Environment Depth:**
   - Find "Experimental Features" section
   - Check **☑ Environment Depth** (for better occlusion quality)
   - Note: This may impact performance slightly

**✅ Verification:**
- OVRCameraRig → OVRManager component
- Quest Features → Depth Submission: ☑ Enabled
- Quest Features → Scene Support: ☑ Enabled

---

### Step 3: Configure MR Utility Kit (Scene API)

**This enables automatic table detection**

1. **Add MRUK Manager (if not already):**
   - In Hierarchy, right-click → Create Empty
   - Rename to: "MRUKManager"
   - Add Component → Search "MRUK" → Add "MRUK" component

2. **Configure MRUK:**
   - In Inspector, find MRUK component
   - Enable **☑ Auto-load Scene on Start**
   - This automatically loads scanned room data

**✅ Verification:**
- MRUKManager GameObject exists
- Has MRUK component
- Auto-load enabled

---

### Step 4: Set Terrain Layer for Occlusion

**Make sure terrain respects depth occlusion**

1. **Create Terrain Layer:**
   - Top menu → Edit → Project Settings → Tags and Layers
   - Under "Layers", find empty slot (e.g., Layer 8)
   - Name it: "Terrain"

2. **Assign Layer to Terrain:**
   - Select Terrain GameObject in Hierarchy
   - In Inspector, find **Layer** dropdown (top-right)
   - Select "Terrain"

3. **Configure Occlusion:**
   - Terrain should now respect depth occlusion automatically
   - No additional settings needed!

---

## 🎬 How It Works In Practice

### Teacher's Workflow:

1. **Start App on Quest:**
   - App loads, detects room/table
   - Terrain appears on detected table

2. **Teacher Zooms In:**
   - Terrain gets larger
   - Parts extending beyond table edges → **Automatically hidden**
   - Only the part on the table is visible

3. **Teacher Rotates:**
   - Terrain spins
   - Different parts go on/off table
   - Occlusion updates in real-time (60fps)

4. **Students See Same Thing:**
   - All students see identical occlusion
   - Synced via network + local Scene API

### What Users See:

```
Physical Table (Top View):
┌─────────────────────┐
│                     │  ← Table edges
│    [Terrain Map]    │  ← Visible part
│  /─────────────\    │
│  │ ████████████ │   │  ← Extended terrain
│  │ ████████████ │   │
└──│─────────────│────┘
   │ ████████████ │     ← This part is HIDDEN by occlusion!
   \─────────────/
```

---

## 🔄 Solution 2: Manual Backup (TerrainBoundsManager)

### When To Use:

Only use this if Meta Quest's automatic occlusion fails:
- Room scan incomplete
- Poor lighting conditions
- Scene API not available
- Need manual bounds control

### Setup:

1. **Add Component to Terrain:**
   - Select Terrain in Hierarchy
   - Add Component → TerrainBoundsManager

2. **Assign References:**
   - Surface Anchor: Drag SurfaceAnchorManager
   - Auto Detect: ☑ Enabled

3. **Choose Mode:**
   - **Warning Only:** Shows red outline (teacher can still move terrain)
   - **Soft Constraint:** Gently pushes terrain back
   - **Hard Constraint:** Blocks movement beyond edges

4. **Enable Visualizer:**
   - Show Bounds Visualizer: ☑ Enabled
   - Shows green/red line around table edges

---

## 📊 Comparison: Automatic vs Manual

| Feature | Quest Auto Occlusion | TerrainBoundsManager |
|---------|---------------------|---------------------|
| **Setup** | 3 checkboxes | Add script + configure |
| **Accuracy** | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐ Good |
| **Performance** | ⭐⭐⭐⭐⭐ Native HW | ⭐⭐⭐⭐ Software |
| **Realism** | ⭐⭐⭐⭐⭐ Real depth | ⭐⭐⭐ Simulated |
| **Works for any surface** | ✅ Yes | ❌ Only rectangles |
| **Requires room scan** | ✅ Yes | ✅ Yes (for auto) |
| **Manual override** | ❌ No | ✅ Yes |

**Recommendation:** Always use automatic occlusion. Keep TerrainBoundsManager as fallback.

---

## 🧪 Testing Occlusion

### In Unity Editor:

**Cannot test occlusion in editor!** It requires:
- Real Quest device
- Physical table
- Room scan data

**What you CAN test in editor:**
- TerrainBoundsManager visualizer (green/red lines)
- Bounds detection logic
- Warning system

### On Meta Quest Device:

1. **Complete Room Setup:**
   - First launch, Quest asks to scan room
   - Point cameras at table
   - Trace table edges by looking around it
   - Quest creates table mesh

2. **Launch Your App:**
   - Terrain appears on detected table
   - Try zooming in (make terrain large)
   - Walk around table
   - **Observe:** Parts beyond edges are clipped!

3. **Verify Occlusion Working:**
   - Zoom terrain to 3x scale (should exceed table)
   - Parts on table = visible
   - Parts beyond edges = hidden
   - Look from different angles - occlusion updates

---

## 🔍 Troubleshooting Occlusion

### Occlusion Not Working:

**1. Check OVRManager Settings:**
- OVRCameraRig → OVRManager → Depth Submission: Must be ☑

**2. Check Room Scan:**
- Quest must have completed room setup
- Table must be detected
- Re-scan room: Settings → Physical Space → Clear Room Data → Re-scan

**3. Check Lighting:**
- Room must be well-lit
- Avoid direct sunlight (confuses sensors)
- Turn on room lights

**4. Check Permissions:**
- App must have Scene permission
- Check in Quest settings → Apps → Your App → Permissions

### Terrain Fully Visible (Not Clipping):

**Possible causes:**
- Depth Submission not enabled
- Room scan incomplete
- Table not detected
- App doesn't have Scene permission

**Solution:**
- Enable Depth Submission in OVRManager
- Re-scan room on Quest
- Grant Scene permission

### Occlusion Too Aggressive:

**If parts that SHOULD be visible are hidden:**
- Terrain might be too close to table surface
- Increase terrain height offset in SurfaceAnchorManager
- Default: 0.05m (5cm), try 0.08m (8cm)

---

## 🎮 User Experience

### What Teacher Sees:

**Good Occlusion (Automatic):**
- Terrain appears to "sit" on physical table
- Seamlessly blends with real world
- Parts beyond table just "disappear" naturally
- Looks like a holographic projection ON the table

**Poor/No Occlusion:**
- Terrain floats in mid-air
- Extends beyond table into void
- Breaks immersion
- Looks fake

### What Students See:

**Everyone sees identical occlusion:**
- Each student's Quest runs Scene API locally
- Same table = same occlusion
- Synced view of where terrain is visible

---

## 📋 Setup Checklist

**For Automatic Occlusion (Use This!):**
- [ ] OVRManager → Depth Submission: ☑
- [ ] OVRManager → Scene Support: ☑
- [ ] MRUK component added to scene
- [ ] User completed Quest room setup
- [ ] App has Scene permission
- [ ] Table detected by Scene API
- [ ] Terrain layer assigned
- [ ] SurfaceAnchorManager configured

**For Manual Fallback (Backup Only):**
- [ ] TerrainBoundsManager added to Terrain
- [ ] SurfaceAnchorManager reference assigned
- [ ] Bounds Mode selected
- [ ] Show Bounds Visualizer: ☑ (for debugging)

---

## 🚀 Recommended Configuration

**For production app:**

```
OVRCameraRig (GameObject)
└─ OVRManager (Component)
    ├─ Depth Submission: ☑ ENABLED
    ├─ Scene Support: ☑ ENABLED
    └─ Environment Depth: ☑ ENABLED (optional)

Terrain (GameObject)
├─ SurfaceAnchorManager (auto-anchors to table)
├─ TerrainInteractionManager (handles zoom/pan/rotate)
└─ TerrainBoundsManager (OPTIONAL backup, set to Disabled or WarningOnly)
```

**Why both?**
- Automatic occlusion handles visual clipping
- TerrainBoundsManager can show warnings to teacher
- Example: "60% of terrain is off table - zoom out"

---

## 📖 Additional Resources

**Meta Documentation:**
- Scene API: https://developer.oculus.com/documentation/unity/unity-scene-overview/
- Depth API: https://developer.oculus.com/documentation/unity/unity-depth-api/
- MR Utility Kit: https://developer.oculus.com/documentation/unity/unity-mr-utility-kit-overview/

**Unity Settings:**
- XR Plugin Management: Oculus provider must be enabled
- Android Platform: Must be selected
- Scene permission: Required in AndroidManifest.xml (Meta XR SDK handles this)

---

## 💡 Pro Tips

### For Best Occlusion:

1. **Good Lighting:** Well-lit room = better depth sensing
2. **Clean Table:** Remove clutter for accurate detection
3. **Re-scan When Needed:** If table moves, re-scan room
4. **Anchor Height:** Keep terrain slightly above table (5-8cm)
5. **Terrain Size:** Start with smaller size, fits on table better

### For Teaching:

1. **Start Zoomed Out:** Full terrain fits on table
2. **Warn Before Zoom:** "I'm going to zoom in - you'll see less area"
3. **Use Markers:** Place pins on visible parts first
4. **Gradual Changes:** Zoom/rotate slowly so students follow

### Performance:

- Automatic occlusion uses hardware acceleration (fast!)
- No FPS impact
- Works even with complex terrain meshes
- Scales to multiple students

---

## 🎓 Summary

**✅ USE THIS (Recommended):**
- Meta Quest 3 automatic MR occlusion
- Enable Depth Submission in OVRManager
- Enable Scene Support in OVRManager
- Zero code required
- Perfect results

**⚠️ BACKUP ONLY:**
- TerrainBoundsManager script (already created)
- Use for warnings/debugging
- Set to "Disabled" or "WarningOnly" by default
- Only enable if automatic occlusion fails

**The automatic method is superior in every way!**

---

**End of Table Occlusion Setup Guide**