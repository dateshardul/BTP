# Quick Start Guide - Multi-User Terrain Interaction

**Last Updated:** 2025-10-22
**Time to Complete:** 30-45 minutes

---

## Prerequisites

- Unity Hub installed
- Unity 2022.3 LTS or newer
- Meta Quest device (or Meta XR Simulator)
- Basic Unity knowledge

---

## Step 1: Create New Unity Project

1. Open Unity Hub
2. Click "New Project"
3. Select "3D (URP)" or "3D Core" template
4. **Project Name:** `MultiUserTerrain`
5. **Location:** Choose your desired folder
6. Click "Create Project"
7. Wait for Unity to initialize

---

## Step 2: Install Required Packages

### A. Install Unity Netcode for GameObjects

1. Open Unity Package Manager: `Window > Package Manager`
2. Click `+` button (top-left)
3. Select "Add package by name..."
4. Enter: `com.unity.netcode.gameobjects`
5. Click "Add"
6. Wait for installation to complete

### B. Install Meta XR Core SDK

**Option 1: Asset Store**
1. Open Asset Store: `Window > Asset Store`
2. Search for "Meta XR Core SDK"
3. Download and import

**Option 2: Manual Download**
1. Download from [Meta Developer Portal](https://developer.oculus.com/downloads/package/unity-integration/)
2. Import package: `Assets > Import Package > Custom Package`
3. Select downloaded `.unitypackage`
4. Click "Import All"

### C. Install TextMeshPro (if prompted)

1. If prompted, click "Import TMP Essentials"
2. Otherwise, it's likely already included

---

## Step 3: Import Scripts

1. In Unity Project window, create folder: `Assets/Scripts`
2. Copy all `.cs` files from `BTP v1\Scripts\` to `Assets/Scripts/`
   - TerrainInteractionManager.cs
   - TerrainInputHandler.cs
   - UserActionIndicator.cs
   - NetworkConnectionManager.cs
   - ConnectionUI.cs
   - PlayerSpawnManager.cs
   - TerrainNetworkSetup.cs

---

## Step 4: Configure Project for Meta Quest

### A. Switch to Android Platform

1. Go to `File > Build Settings`
2. Select "Android"
3. Click "Switch Platform" (wait for reimport)

### B. Configure Player Settings

1. In Build Settings, click "Player Settings"
2. Configure:
   - **Company Name:** Your name
   - **Product Name:** MultiUserTerrain
   - **Other Settings:**
     - Minimum API Level: Android 10.0 (API 29)
     - Target API Level: Android 12.0 or newer
     - Scripting Backend: IL2CPP
     - Target Architectures: ARM64 ✓

### C. Configure XR Settings

1. Go to `Edit > Project Settings > XR Plug-in Management`
2. Click "Install XR Plugin Management" if needed
3. Select "Android" tab
4. Enable "Oculus" (or "Meta Quest")

---

## Step 5: Create Scene

### A. Set Up Basic Scene

1. Create new scene: `File > New Scene > Basic (URP or Built-in)`
2. Save as: `Assets/Scenes/MultiUserTerrain.unity`
3. Delete default "Main Camera"

### B. Add Network Manager

1. Create empty GameObject: `Right-click > Create Empty`
2. Rename to "NetworkManager"
3. Add Component: `NetworkManager` (Unity Netcode)
4. Add Component: `Unity Transport`
5. In NetworkManager component:
   - Transport: Drag `Unity Transport` component here

### C. Add Connection Manager

1. Create empty GameObject: "ConnectionManager"
2. Add Component: `NetworkConnectionManager`
3. In NetworkConnectionManager:
   - IP Address: 127.0.0.1 (for local testing)
   - Port: 7777

### D. Add OVR Camera Rig

1. In Project window, search for "OVRCameraRig"
2. Drag OVRCameraRig prefab into scene
3. Position at (0, 0, 0)

### E. Add Terrain

1. Create: `3D Object > Plane`
2. Rename to "Terrain"
3. Position: (0, 1, 0) - at waist height
4. Scale: (0.3, 0.3, 0.3) - tabletop size
5. Add Component: `NetworkObject`
6. Add Component: `TerrainInteractionManager`
7. Configure TerrainInteractionManager:
   - Terrain Transform: Drag "Terrain" itself here
   - Zoom Speed: 0.5
   - Pan Speed: 1.0
   - Rotate Speed: 50
   - Zoom Limits: Min 0.5, Max 3.0
   - Pan Radius: 5.0
   - User Colors: Size 4, add colors (Red, Green, Blue, Yellow)

### F. Add Player Spawn Manager

1. Create empty GameObject: "PlayerSpawnManager"
2. Add Component: `NetworkObject`
3. Add Component: `PlayerSpawnManager`
4. Configure PlayerSpawnManager:
   - Terrain Center: Drag "Terrain" here
   - Spawn Radius: 2.0
   - Spawn Height: 1.6

---

## Step 6: Create Player Prefab

1. Create empty GameObject: "Player"
2. Add Component: `NetworkObject`
3. Add child GameObject: "InputHandler"
4. On InputHandler, add Component: `TerrainInputHandler`
5. In TerrainInputHandler:
   - Terrain Manager: (leave empty, will assign at runtime)
6. Add Component: `TerrainNetworkSetup`
7. Drag "Player" to Project window to create prefab
8. Delete from scene

---

## Step 7: Configure NetworkManager

1. Select "NetworkManager" GameObject
2. In NetworkManager component:
   - **Player Prefab:** Drag Player prefab here
   - **Network Prefabs List:**
     - Add "Terrain" (from scene)
     - Add "Player" (prefab)

---

## Step 8: Create Simple Connection UI

### A. Create Canvas

1. Create: `UI > Canvas`
2. Set Canvas Scaler:
   - UI Scale Mode: Scale With Screen Size
   - Reference Resolution: 1920 x 1080

### B. Create Buttons

1. Create: `UI > Button - TextMeshPro` (3 times)
2. Rename them:
   - "HostButton"
   - "ClientButton"
   - "DisconnectButton"
3. Position vertically in center of screen
4. Update button texts:
   - HostButton: "Start as Host"
   - ClientButton: "Join as Client"
   - DisconnectButton: "Disconnect"

### C. Create Input Fields

1. Create: `UI > Input Field - TextMeshPro` (2 times)
2. Rename:
   - "IPInputField"
   - "PortInputField"
3. Position above buttons
4. Set placeholder text:
   - IPInputField: "Enter IP Address"
   - PortInputField: "Enter Port"

### D. Create Status Text

1. Create: `UI > Text - TextMeshPro`
2. Rename: "StatusText"
3. Position at top of canvas
4. Set text: "Ready to connect"

### E. Connect UI to Script

1. Create empty GameObject: "UIManager"
2. Add Component: `ConnectionUI`
3. In ConnectionUI component:
   - Host Button: Drag HostButton
   - Client Button: Drag ClientButton
   - Disconnect Button: Drag DisconnectButton
   - IP Input Field: Drag IPInputField
   - Port Input Field: Drag PortInputField
   - Status Text: Drag StatusText
   - Connection Manager: Drag ConnectionManager
   - Default IP: 127.0.0.1
   - Default Port: 7777

---

## Step 9: Test in Unity Editor

### Single Player Test

1. Click Play
2. Click "Start as Host"
3. You should see the terrain in VR (if Meta Quest is connected)

### Multi-Player Test (Two Instances)

1. Go to `Edit > Project Settings > Player`
2. Under "Resolution and Presentation":
   - Check "Run in Background"
3. Build the project: `File > Build Settings > Build`
4. Run the built executable
5. In one instance: Click "Start as Host"
6. In other instance: Enter host IP, click "Join as Client"

---

## Step 10: Build to Quest

1. Connect Meta Quest via USB
2. Enable Developer Mode on Quest
3. Go to `File > Build Settings`
4. Click "Add Open Scenes"
5. Click "Build and Run"
6. Choose save location
7. Wait for build and deployment

---

## Troubleshooting

### "NetworkManager not found"
- Make sure NetworkManager GameObject exists in scene
- Ensure it has NetworkManager component

### "OVRCameraRig not found"
- Meta XR SDK not installed properly
- Re-import Meta XR Core SDK

### "Scripts have compile errors"
- Ensure Unity Netcode package is installed
- Ensure Meta XR SDK is installed
- Check Console for specific errors

### Terrain doesn't move
- Check TerrainInputHandler has reference to TerrainInteractionManager
- Verify NetworkObject component is on Terrain
- Ensure you're the owner (started as host or granted authority)

### Players can't connect
- Check both devices on same network
- Verify IP address is correct
- Check firewall settings
- Port 7777 must be open

---

## Next Steps

Once basic functionality works:

1. **Add Visual Indicators**
   - Create manipulation indicator prefab
   - Add glow/ring around terrain when manipulated

2. **Improve UI**
   - Add player list
   - Show who's manipulating terrain
   - Add reset button

3. **Add Advanced Features**
   - Gesture recognition
   - Multiple terrains
   - Save/load terrain states

4. **Optimize**
   - Test with 4+ users
   - Optimize network traffic
   - Improve visual feedback

---

## Useful Commands

### Unity Editor
- Play: `Ctrl/Cmd + P`
- Build: `Ctrl/Cmd + B`
- Console: `Ctrl/Cmd + Shift + C`

### Meta Quest (via ADB)
- Install APK: `adb install path/to/app.apk`
- View logs: `adb logcat`
- Uninstall: `adb uninstall com.yourcompany.multiuserterrain`

---

## Resources

- Unity Netcode Docs: https://docs-multiplayer.unity3d.com/netcode/current/about/
- Meta XR SDK Docs: https://developer.oculus.com/documentation/unity/
- Unity Manual: https://docs.unity3d.com/Manual/index.html

---

**For detailed setup instructions, see `unity_project_setup.md`**