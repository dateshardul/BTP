# Implementation Memo - Multi-User Terrain Interaction

**Project:** Multi-User Terrain Interaction System
**Last Updated:** 2025-10-22
**Status:** Core scripts created, Unity integration pending

---

## File Structure & Contents

### Planning Files
- `multi_user_terrain_interaction_plan.md` - Complete implementation plan (updated for Unity Netcode)
- `status_report.md` - Progress tracking (updated after each completed task)
- `implementation_memo.md` - This file - important notes and reminders
- `unity_project_setup.md` - Detailed Unity setup guide

### Created Scripts (14 Total) ✅ ALL COMPLETE

**Core Terrain (3):**
- `Scripts/TerrainInteractionManager.cs` - ✓ Unity Netcode version
- `Scripts/TerrainInputHandler.cs` - ✓ XR input handling
- `Scripts/UserActionIndicator.cs` - ✓ Visual feedback

**Networking (4):**
- `Scripts/NetworkConnectionManager.cs` - ✓ Host/Client management
- `Scripts/ConnectionUI.cs` - ✓ Connection UI
- `Scripts/PlayerSpawnManager.cs` - ✓ Spawn around terrain
- `Scripts/TerrainNetworkSetup.cs` - ✓ Auto-connect helper

**Teacher Interaction (7):**
- `Scripts/TeacherControlMode.cs` - ✓ Role-based access
- `Scripts/PointerBasedTerrainController.cs` - ✓ IMPROVED UX controller
- `Scripts/HandGestureTerrainController.cs` - ✓ Hand tracking support
- `Scripts/SurfaceAnchorManager.cs` - ✓ Table/ground anchoring
- `Scripts/AnnotationSystem.cs` - ✓ Marker management
- `Scripts/MarkerPin.cs` - ✓ Google Maps-style pins
- `Scripts/TerrainBoundsManager.cs` - ✓ BACKUP manual table clipping

### Documentation Files (8 Total):
- `planning/unity_project_setup.md` - ✓ Beginner Unity guide
- `planning/teacher_interaction_guide.md` - ✓ Control instructions
- `planning/table_occlusion_setup.md` - ✓ Edge clipping guide
- `planning/implementation_memo.md` - ✓ This file
- `planning/status_report.md` - ✓ Progress tracking
- `planning/quick_start_guide.md` - ✓ Quick tutorial
- `planning/multi_user_terrain_interaction_plan.md` - ✓ Original plan
- `planning/IMPLEMENTATION_COMPLETE_SUMMARY.md` - ✓ Final summary
- `README.md` - ✓ Project overview

---

## Key Architecture Components

### Core Scripts (CREATED)
1. **TerrainInteractionManager.cs**
   - Location: `C:\Users\shard\OneDrive\Desktop\IITJ\BTP\BTP v1\Scripts\`
   - Handles zoom, pan, rotate operations
   - Network synchronization using Unity Netcode NetworkBehaviour
   - Authority management to prevent conflicts
   - Adapted from Photon Fusion to Unity Netcode

2. **TerrainInputHandler.cs**
   - Location: `C:\Users\shard\OneDrive\Desktop\IITJ\BTP\BTP v1\Scripts\`
   - Processes controller and hand tracking input
   - Converts input to terrain manipulation calls
   - Supports both XR controllers and OVR hand tracking
   - No changes needed from original plan

3. **UserActionIndicator.cs**
   - Location: `C:\Users\shard\OneDrive\Desktop\IITJ\BTP\BTP v1\Scripts\`
   - Visual feedback system
   - Shows what other users are doing
   - Networked indicators and text labels
   - Adapted from Photon Fusion to Unity Netcode

### Networking Framework Decision
**DECIDED: Unity Netcode for GameObjects**
- ✓ Free and open source
- ✓ Unity-native integration
- ✓ Active development and support
- Voice chat: Deferred (not needed initially, can add later)

---

## Key Technical Concepts (Unity Netcode)

### State Authority Management
- Only one user can manipulate terrain at a time
- Uses `IsOwner` property to control who can make changes
- ServerRPCs (`RequestStateAuthorityServerRpc`, `ReleaseStateAuthorityServerRpc`) manage handoffs
- `NetworkObject.ChangeOwnership()` transfers control between clients

### Networked Properties (Unity Netcode Adaptation)
**Original Photon Fusion:**
```csharp
[Networked] public Vector3 NetworkedPosition { get; set; }
```

**Unity Netcode Version:**
```csharp
private NetworkVariable<Vector3> networkedPosition = new NetworkVariable<Vector3>(Vector3.zero);
private NetworkVariable<Quaternion> networkedRotation = new NetworkVariable<Quaternion>(Quaternion.identity);
private NetworkVariable<float> networkedScale = new NetworkVariable<float>(1f);
private NetworkVariable<bool> isBeingManipulated = new NetworkVariable<bool>(false);
private NetworkVariable<ulong> currentManipulatorId = new NetworkVariable<ulong>(0);
```

**Key Differences from Photon Fusion:**
- `[Networked]` properties → `NetworkVariable<T>`
- `PlayerRef` → `ulong` (client ID)
- `[Rpc]` attributes → `[ServerRpc]` and `[ClientRpc]`
- `Object.HasStateAuthority` → `IsOwner`
- `FixedUpdateNetwork()` → `OnValueChanged` callbacks

### Interaction Types
1. **Zoom**: Pinch gesture or controller trigger
2. **Pan**: Single-hand grab and move
3. **Rotate**: Two-handed twist or controller rotation

---

## Critical Dependencies

### Required SDKs/Packages
- Unity Netcode for GameObjects (com.unity.netcode.gameobjects)
- Meta XR Core SDK (v78+)
- Multiplayer Building Blocks
- Meta XR Platform SDK (for user invites)
- TextMeshPro (usually included with Unity)

### Unity Components Needed
- NetworkManager (Unity Netcode)
- NetworkObject (on all networked GameObjects)
- OVRCameraRig (Meta XR SDK)
- OVRHand components for hand tracking
- InputDevice for XR controller input
- LineRenderer for visual indicators
- TextMeshPro for user action text

### Unity Prefabs to Create
- NetworkedTerrain.prefab (terrain with NetworkObject + TerrainInteractionManager)
- Player.prefab (player with NetworkObject + input handling)
- ManipulationIndicator.prefab (visual ring around terrain)
- UserActionIndicator.prefab (shows user actions)

---

## Implementation Strategy

### Testing Approach
1. **Single User Testing**: Verify basic terrain manipulation works
2. **Meta XR Simulator**: Test multiplayer functionality
3. **Network Sync Validation**: Ensure all users see same state
4. **Authority Testing**: Verify only one user can manipulate at a time

### Visual Feedback Requirements
- Color-coded indicators per user
- Clear action labels (Zooming, Panning, Rotating)
- Hand position indicators
- Manipulation boundaries/limits

---

## Important Constraints

### Interaction Limits (from TerrainInteractionManager)
- Zoom limits: 0.5x to 3x scale
- Pan radius: 5 units maximum
- Authority: One user at a time

### Performance Considerations (Unity Netcode)
- NetworkVariable changes automatically sync to all clients
- OnValueChanged callbacks update visuals in real-time
- ServerRPCs require ownership (use RequireOwnership = true/false)
- Consider interpolation for smooth visuals
- Network tick rate: 30 Hz (configurable in NetworkManager)

---

## Phase Dependencies
- Phase 1 must complete before Phase 2 can begin
- Friend will handle most of Phase 1 SDK setup
- Phase 2 requires Unity project structure to be established
- Networking framework choice affects all subsequent phases

---

## Quick Reference - Key Methods

### TerrainInteractionManager (Unity Netcode)
**Public Methods:**
- `StartManipulation()` - Begin terrain interaction
- `EndManipulation()` - End terrain interaction
- `ZoomTerrain(float zoomDelta)` - Scale terrain
- `PanTerrain(Vector3 handPosition)` - Move terrain
- `RotateTerrain(float rotationDelta)` - Rotate terrain
- `RotateTerrainTwoHanded(Vector3 hand1, Vector3 hand2)` - Two-hand rotation
- `PinchZoom(Vector3 hand1, Vector3 hand2, bool isInitial)` - Two-hand zoom
- `ResetTerrainServerRpc()` - Reset to default state

**Authority Management (ServerRPCs):**
- `RequestStateAuthorityServerRpc()` - Request control (internal)
- `ReleaseStateAuthorityServerRpc()` - Release control (internal)

**Property Access:**
- `IsOwner` - Check if local client owns terrain
- `NetworkedPosition` - Get current position
- `NetworkedRotation` - Get current rotation
- `NetworkedScale` - Get current scale
- `IsBeingManipulated` - Check if someone is manipulating
- `CurrentManipulator` - Get client ID of current manipulator

### UserActionIndicator (Unity Netcode)
- `UpdateHandPosition(Vector3 handPos)` - Update hand position indicator
- `UpdateActionType(int actionType)` - Update action type (0=none, 1=zoom, 2=pan, 3=rotate)

### NetworkVariable Usage Pattern
```csharp
// Subscribe to changes in OnNetworkSpawn
networkedPosition.OnValueChanged += OnPositionChanged;

// Update value (only owner/server can do this)
networkedPosition.Value = newPosition;

// Access current value (anyone can read)
Vector3 pos = networkedPosition.Value;
```

---

## Photon Fusion → Unity Netcode Migration Notes

### Class Inheritance
- `Fusion.NetworkBehaviour` → `Unity.Netcode.NetworkBehaviour`

### Properties
- `[Networked] Type Property { get; set; }` → `NetworkVariable<Type> property`

### RPCs
- `[Rpc(RpcSources.All, RpcTargets.StateAuthority)]` → `[ServerRpc(RequireOwnership = false)]`
- `RpcInfo info` parameter → `ServerRpcParams rpcParams`
- `info.Source` → `rpcParams.Receive.SenderClientId`

### Authority
- `Object.HasStateAuthority` → `IsOwner`
- `Object.AssignInputAuthority(player)` → `NetworkObject.ChangeOwnership(clientId)`
- `Object.RemoveInputAuthority()` → `NetworkObject.RemoveOwnership()`

### Updates
- `FixedUpdateNetwork()` → `OnValueChanged` callbacks on NetworkVariables

### Player References
- `PlayerRef` → `ulong` (client ID)
- `PlayerRef.None` → `0`

---

---

## IMPROVED UX Control Scheme (Meta Quest 3)

### Controller Layout:
| Input | Action | Description |
|-------|--------|-------------|
| **Button A** | Place Marker | Instant pin placement with haptic feedback |
| **Button B** | Undo Marker | Remove last placed marker |
| **Grip + Move** | Pan | Drag terrain along table surface |
| **Trigger + Move** | Zoom | Push=zoom in, Pull=zoom out (centered on pointer) |
| **Trigger + Stick ←→** | Rotate | Spin terrain around pointer point |

### Hand Gestures (Alternative):
| Gesture | Action | Description |
|---------|--------|-------------|
| **Single Pinch + Move** | Pan | Grab and drag terrain |
| **Two-Hand Pinch Spread** | Zoom | Spread hands = zoom in, together = zoom out |
| **Two-Hand Twist** | Rotate | Twist hands to rotate terrain |
| **Point + Dwell (1s)** | Annotate | Hold point steady to place marker |

### Pointer Colors:
- **Cyan:** Idle
- **Blue:** Zoom active (Trigger only)
- **Green:** Pan active (Grip pressed)
- **Yellow:** Rotate active (Trigger + Thumbstick)
- **Red:** Annotate active (Button A or pointing)

---

## Teacher vs Student Roles

### Teacher Capabilities:
- ✅ Controller pointer with ray
- ✅ Hand gesture tracking
- ✅ Zoom/Pan/Rotate terrain
- ✅ Place/remove markers
- ✅ See control UI
- ✅ Network authority for terrain

### Student Capabilities:
- ✅ View terrain in real-time
- ✅ See all markers
- ✅ Move around table physically
- ❌ Cannot manipulate terrain
- ❌ No pointer ray
- ❌ No control UI

---

---

## Table Edge Occlusion

### Primary Method: Meta Quest 3 Automatic (Recommended)
**Setup in Unity:**
1. OVRCameraRig → OVRManager → Quest Features:
   - ☑ Depth Submission (CRITICAL!)
   - ☑ Scene Support
2. That's it! Quest automatically hides terrain parts beyond table

**How it works:**
- Quest's depth cameras scan environment
- Detects table edges automatically
- Hides virtual objects behind real objects
- Zero performance cost (hardware accelerated)

### Backup Method: TerrainBoundsManager (Manual)
**Use only if automatic fails:**
- Add TerrainBoundsManager component to Terrain
- Provides visual warnings when terrain exceeds bounds
- Optional constraints (Warning/Soft/Hard modes)
- See `table_occlusion_setup.md` for details

---

## Next Action Items

**COMPLETED:**
1. ✓ Decide on networking framework (Unity Netcode chosen)
2. ✓ Create all 14 scripts
3. ✓ Design IMPROVED UX control scheme
4. ✓ Add hand gesture support
5. ✓ Add table edge occlusion (automatic + backup)
6. ✓ Create comprehensive Unity setup documentation (8 docs)

**TODO (Unity Integration):**
7. **Follow unity_project_setup.md** to create Unity project
8. Install Unity Netcode for GameObjects package
9. Install Meta XR SDK
10. Enable Depth Submission in OVRManager
11. Set up scene with all GameObjects
12. Create MarkerPin prefab
13. Add MarkerPin to Network Prefabs List
14. Test single-user functionality
15. Test multi-user synchronization with teacher+students
16. Test table edge occlusion
17. Build to Meta Quest 3 device