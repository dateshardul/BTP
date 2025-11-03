# Multi-User Terrain Interaction System

A collaborative mixed reality application for Meta Quest that allows multiple users to simultaneously view and interact with a shared 3D terrain model using hand tracking and controller inputs.

**Status:** Scripts Complete (45% Overall) - Ready for Unity Integration
**Last Updated:** 2025-10-22

---

## Project Overview

This system enables multiple users in a shared VR/MR space to:
- View the same terrain model simultaneously
- Zoom, pan, and rotate the terrain collaboratively
- See visual indicators of what other users are doing
- Use both hand tracking (pinch gestures) and controllers

**Key Features:**
- Multi-user synchronization via Unity Netcode
- Gesture-based interaction (pinch, grab, rotate)
- Authority management (prevents conflicts)
- Visual feedback system
- Automatic player spawn positioning

---

## Project Structure

```
BTP v1/
├── Scripts/                          # All C# scripts (7 files)
│   ├── TerrainInteractionManager.cs # Core terrain manipulation & networking
│   ├── TerrainInputHandler.cs       # XR input handling (controllers + hands)
│   ├── UserActionIndicator.cs       # Visual feedback for user actions
│   ├── NetworkConnectionManager.cs  # Host/Client connection management
│   ├── ConnectionUI.cs              # UI for network connection
│   ├── PlayerSpawnManager.cs        # Spawn players around terrain
│   └── TerrainNetworkSetup.cs       # Auto-connect components helper
│
└── planning/                         # Documentation & guides
    ├── multi_user_terrain_interaction_plan.md  # Original implementation plan
    ├── status_report.md              # Progress tracking
    ├── implementation_memo.md        # Technical notes & quick reference
    ├── unity_project_setup.md        # Comprehensive Unity setup guide
    └── quick_start_guide.md          # Step-by-step getting started

```

---

## Quick Start

### Option 1: Follow Quick Start Guide
See `planning/quick_start_guide.md` for a step-by-step walkthrough (30-45 minutes)

### Option 2: Follow Detailed Setup
See `planning/unity_project_setup.md` for comprehensive instructions

### Minimum Steps:
1. Create Unity 2022.3 LTS project
2. Install Unity Netcode for GameObjects package
3. Install Meta XR Core SDK (v78+)
4. Copy all scripts from `Scripts/` folder to `Assets/Scripts/`
5. Follow scene setup in quick start guide
6. Build to Meta Quest

---

## Technologies Used

- **Unity 2022.3 LTS** - Game engine
- **Unity Netcode for GameObjects** - Networking framework (free)
- **Meta XR Core SDK** - VR/MR functionality
- **Meta Quest** - Target hardware platform
- **C#** - Programming language

---

## Scripts Overview

### Core Terrain Interaction (3 scripts)

**TerrainInteractionManager.cs**
- Manages terrain position, rotation, and scale
- Network synchronization using Unity Netcode
- Authority management (one user at a time)
- Public methods: `StartManipulation()`, `ZoomTerrain()`, `PanTerrain()`, `RotateTerrain()`

**TerrainInputHandler.cs**
- Processes XR controller and hand tracking input
- Supports both Meta Quest controllers and hand tracking
- Converts gestures to terrain manipulation commands
- Handles single-hand pan and two-hand zoom/rotate

**UserActionIndicator.cs**
- Shows visual indicators of user actions
- Displays hand position and action type (zoom/pan/rotate)
- Network synchronized across all clients

### Networking & Setup (4 scripts)

**NetworkConnectionManager.cs**
- Manages network sessions (Host/Client/Server)
- Handles connection/disconnection events
- Provides public methods: `StartHost()`, `StartClient()`, `Disconnect()`

**ConnectionUI.cs**
- Simple UI for connecting to network sessions
- Input fields for IP address and port
- Buttons for Host/Client/Disconnect
- Status text display

**PlayerSpawnManager.cs**
- Spawns players in a circle around the terrain
- Automatic position assignment (N, E, S, W)
- Players face the terrain center
- Gizmos for visualization in editor

**TerrainNetworkSetup.cs**
- Helper script to auto-connect components
- Finds and links TerrainInputHandler to TerrainInteractionManager
- Optional manual connection method

---

## Implementation Progress

### ✅ Completed (45%)

**Phase 1: Setup (50%)**
- [x] Networking framework decision (Unity Netcode)
- [ ] Install Unity Netcode package
- [ ] Install Meta XR SDK

**Phase 2: Implementation (80%)**
- [x] All core scripts created (7 files)
- [x] Terrain interaction system
- [x] Input handling system
- [x] Network connection system
- [x] Player spawn system
- [ ] Create Unity scene
- [ ] Create prefabs
- [ ] Test basic functionality

**Phase 4: Advanced Features (25%)**
- [x] Player spawn system
- [ ] Gesture recognition
- [ ] User management

### ⏳ To Do (55%)

**Phase 3: Multi-User Testing (0%)**
- [ ] Test with Meta XR Simulator
- [ ] Verify network synchronization
- [ ] Test authority management
- [ ] Validate visual feedback

**Unity Integration**
- [ ] Create Unity project
- [ ] Import all packages
- [ ] Set up scene
- [ ] Create prefabs
- [ ] Build to Quest device

---

## Network Architecture

### Authority Model
- **Server-authoritative:** All state changes go through server
- **Ownership-based:** Only terrain owner can manipulate it
- **RPC pattern:** Clients request authority via ServerRPCs

### Synchronization
- **NetworkVariables:** Auto-sync terrain position, rotation, scale
- **OnValueChanged:** Real-time updates to all clients
- **ServerRPCs:** State changes and authority requests

### Data Flow
```
Client A (Owner)
    ↓ [User Input]
TerrainInputHandler
    ↓ [Manipulation Request]
TerrainInteractionManager
    ↓ [ServerRPC]
Server
    ↓ [NetworkVariable Update]
All Clients (including A)
    ↓ [OnValueChanged callback]
Terrain Transform Updated
```

---

## Key Concepts

### Interaction Types
1. **Zoom:** Pinch gesture or controller trigger (0.5x - 3x scale)
2. **Pan:** Single-hand grab and move (5 unit radius limit)
3. **Rotate:** Two-hand twist or controller rotation

### Authority Management
- Only one user can manipulate terrain at a time
- Authority automatically released when user lets go
- Visual indicators show who has control

### Input Methods
- **Controllers:** Grip button to grab, trigger to zoom
- **Hand Tracking:** Pinch gesture to interact
- Both methods work simultaneously

---

## Configuration

### TerrainInteractionManager Settings
- Zoom Speed: 0.5 (adjustable)
- Pan Speed: 1.0 (adjustable)
- Rotate Speed: 50 (adjustable)
- Zoom Limits: 0.5x to 3.0x scale
- Pan Radius: 5 units maximum

### Network Settings
- Default IP: 127.0.0.1 (localhost)
- Default Port: 7777
- Network Tick Rate: 30 Hz
- Transport: Unity Transport (UDP)

### Spawn Settings
- Spawn Radius: 2 meters around terrain
- Spawn Height: 1.6 meters (eye level)
- Spawn Angles: 0°, 90°, 180°, 270° (N, E, S, W)

---

## Requirements

### Development
- Unity 2022.3 LTS or newer
- Unity Netcode for GameObjects package
- Meta XR Core SDK (v78+)
- TextMeshPro (included with Unity)

### Target Platform
- Meta Quest 2/3/Pro
- Android API Level 29+ (Android 10.0+)
- IL2CPP scripting backend
- ARM64 architecture

### Network
- Local network or internet connection
- Port 7777 open (configurable)
- Same network for all participants

---

## Testing

### Local Testing
1. Build and run on Meta Quest
2. Use Meta XR Simulator in Unity Editor
3. One device as Host, simulator as Client

### Multi-Device Testing
1. Build to two+ Quest devices
2. Connect all to same network
3. One device starts as Host
4. Others join as Clients

---

## Troubleshooting

**Scripts won't compile:**
- Install Unity Netcode for GameObjects package
- Install Meta XR Core SDK
- Check Unity version (2022.3 LTS+)

**Terrain doesn't respond to input:**
- Verify TerrainInputHandler is connected to TerrainInteractionManager
- Check NetworkObject component on terrain
- Ensure you're the owner (Host or granted authority)

**Can't connect to host:**
- Verify both devices on same network
- Check IP address is correct
- Ensure port 7777 is open
- Check firewall settings

---

## Documentation

- **`quick_start_guide.md`** - Step-by-step Unity setup (START HERE)
- **`unity_project_setup.md`** - Comprehensive setup details
- **`implementation_memo.md`** - Technical reference & API docs
- **`status_report.md`** - Current progress tracking
- **`multi_user_terrain_interaction_plan.md`** - Original design doc

---

## Next Steps

1. **Create Unity Project**
   - Use Unity 2022.3 LTS
   - 3D template (URP or Core)

2. **Install Packages**
   - Unity Netcode for GameObjects
   - Meta XR Core SDK

3. **Import Scripts**
   - Copy all 7 scripts to Assets/Scripts/

4. **Follow Quick Start Guide**
   - See `planning/quick_start_guide.md`
   - 30-45 minutes to working prototype

5. **Test & Iterate**
   - Test single user functionality
   - Test multi-user synchronization
   - Adjust speeds and limits as needed

---

## Future Enhancements

- [ ] Voice chat integration
- [ ] Multiple terrain support
- [ ] Annotation/markup tools
- [ ] Save/load terrain states
- [ ] Advanced gesture recognition
- [ ] Mobile device spectator mode
- [ ] Recording/playback of sessions

---

## Credits

**Framework:** Based on Meta's Shared Activities in Mixed Reality Motif
**Networking:** Unity Netcode for GameObjects
**Platform:** Meta Quest + Meta XR SDK

**Developed for:** Bachelor's Thesis Project (BTP)
**Institution:** IITJ

---

## License

[Add your license information here]

---

## Contact

[Add your contact information here]

---

**For questions or issues, check the documentation in the `planning/` folder first!**