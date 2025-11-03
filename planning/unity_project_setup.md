# Unity Project Setup Guide (Beginner-Friendly)

**Last Updated:** 2025-10-22
**Target Unity Version:** 2022.3.62f2 (Your installed version)
**Platform:** Meta Quest (Android)
**Difficulty:** Beginner (No Unity knowledge required)
**Time:** 1-2 hours

---

## 🎮 Unity Basics for Complete Beginners

### What is Unity?
Unity is a game engine (like a sophisticated 3D modeling + programming tool combined). Think of it as:
- **Scene** = A 3D world/level (like a stage)
- **GameObject** = Any object in the 3D world (like actors on stage)
- **Component** = Abilities/behaviors attached to GameObjects (like scripts for actors)
- **Inspector** = Properties panel where you configure things (on the right side)
- **Hierarchy** = List of all GameObjects in your scene (on the left side)
- **Project** = Your file browser for assets (bottom panel)

### Unity Interface Layout (When you open Unity, you'll see):
```
┌─────────────────────────────────────────────────────────┐
│  Menu Bar: File, Edit, Assets, GameObject, etc.        │
├──────────┬──────────────────────────────┬──────────────┤
│          │                              │              │
│ Hierarchy│        Scene View            │  Inspector   │
│  (Left)  │     (Center/Main view)       │   (Right)    │
│          │    Your 3D world             │  Properties  │
│          │                              │              │
├──────────┴──────────────────────────────┤              │
│                                         │              │
│         Project (Bottom)                │              │
│      Your files & assets                │              │
│                                         │              │
└─────────────────────────────────────────┴──────────────┘
```

### Key Unity Concepts You Need to Know:

1. **GameObject** = Everything in your 3D world
   - Like boxes, characters, cameras, lights, invisible managers
   - Created via: Right-click Hierarchy > Create Empty (or 3D Object > ...)

2. **Component** = Features you attach to GameObjects
   - Like scripts, physics, rendering, networking
   - Added via: Select GameObject > Inspector > Add Component button

3. **Prefab** = A reusable template/blueprint
   - Like a cookie cutter - make one, create many
   - Created by: Dragging GameObject from Hierarchy to Project folder

4. **Script** = C# code files (like the ones we created)
   - They control behavior and logic
   - Attached to GameObjects as Components

5. **Inspector** = Settings panel
   - Shows properties of selected GameObject
   - Where you set values, drag-and-drop references, etc.

6. **Scene** = Your 3D world file
   - Saved as .unity file
   - Can have multiple scenes (like levels in a game)

---

## 📋 Before You Start - Checklist

- [x] Unity 2022.3.62f2 installed
- [ ] Meta Quest device (or will use simulator)
- [ ] USB cable for Quest connection
- [ ] All scripts from `BTP v1\Scripts\` folder ready to copy
- [ ] About 1-2 hours of uninterrupted time

---

## 🚀 Step 1: Create Your First Unity Project

### 1.1 Open Unity Hub
- Unity Hub is the launcher (you should see it after installation)
- It manages different Unity versions and projects

### 1.2 Create New Project
1. Click the **"New project"** button (top-right or center)
2. You'll see a list of templates:
   - Select **"3D Core"** template (NOT 3D URP, NOT 2D)
   - This is a standard 3D project template

3. Configure project settings:
   - **Project name:** `MultiUserTerrain` (no spaces issues, but spaces are ok)
   - **Location:** Choose a folder (like `C:\Unity Projects\`)
   - **Unity version:** Should show 2022.3.62f2

4. Click **"Create project"** (bottom-right)
5. **Wait 2-5 minutes** - Unity is setting up your project
   - You'll see a loading screen
   - First time setup takes longer

### 1.3 Unity Editor Opens
You should now see the Unity Editor with:
- A sample scene with a camera and light
- Empty 3D view in the center
- Hierarchy on the left showing "Main Camera" and "Directional Light"
- Inspector on the right
- Project panel at bottom showing "Assets" folder

**✅ Checkpoint:** If you see this layout, you're ready for Step 2!

---

## 📦 Step 2: Install Required Packages

**What are packages?** Think of them as plugins or extensions that add new features to Unity.

### 2.1 Install Unity Netcode for GameObjects (Networking)

**What it does:** Allows multiple users to connect and sync data over network

**How to install:**

1. **Open Package Manager:**
   - Top menu bar → Click `Window` → Select `Package Manager`
   - A new window opens showing available packages

2. **Add Package by Name:**
   - Look for a **`+`** button (top-left corner of Package Manager window)
   - Click it, you'll see a dropdown menu
   - Select **"Add package by name..."**

3. **Enter Package Name:**
   - A small dialog box appears with two fields
   - In the **Name** field, type exactly: `com.unity.netcode.gameobjects`
   - Leave **Version** field empty (uses latest)
   - Click **"Add"** button

4. **Wait for Installation:**
   - You'll see "Installing..." at bottom
   - Takes 1-2 minutes
   - When done, you'll see "Netcode for GameObjects" in the package list

**✅ Verification:** In Package Manager, find "Netcode for GameObjects" with a checkmark.

---

### 2.2 Install Meta XR Core SDK (Quest VR Support)

**What it does:** Adds Meta Quest headset support and hand tracking

**Installation Method 1: Asset Store (Recommended for beginners)**

1. **Open Asset Store:**
   - Top menu → `Window` → `Asset Store`
   - Or visit: https://assetstore.unity.com/ in your browser

2. **Search for Meta XR SDK:**
   - In search bar, type: `Meta XR Core SDK`
   - Look for official Meta package (by "Meta")

3. **Download & Import:**
   - Click on the package
   - Click "Add to My Assets" (if first time)
   - Click "Import" or "Download"
   - Unity will download it (may take 5-10 minutes - it's large!)
   - An import dialog will appear showing all files
   - Click "Import" button (bottom-right) to import everything

4. **Wait for Import:**
   - This can take 5-15 minutes
   - You'll see progress bar at bottom of Unity
   - Console may show some warnings (usually safe to ignore)

**Installation Method 2: Manual Download (Alternative)**

1. Visit: https://developer.oculus.com/downloads/package/unity-integration/
2. Download the `.unitypackage` file
3. In Unity: `Assets` → `Import Package` → `Custom Package`
4. Select the downloaded file
5. Click "Import" in the dialog

**✅ Verification:** In Project panel, you should see a new folder called "Oculus" or "Meta"

---

### 2.3 Import Scripts into Unity

**Now we'll bring in the C# scripts we created earlier**

1. **Create Scripts Folder:**
   - In **Project** panel (bottom), you should see "Assets" folder
   - Right-click on "Assets"
   - Select `Create` → `Folder`
   - Name it **`Scripts`** (exactly, case matters!)

2. **Copy Your Scripts:**
   - Open Windows File Explorer
   - Navigate to: `C:\Users\shard\OneDrive\Desktop\IITJ\BTP\BTP v1\Scripts\`
   - Select ALL 7 `.cs` files:
     - TerrainInteractionManager.cs
     - TerrainInputHandler.cs
     - UserActionIndicator.cs
     - NetworkConnectionManager.cs
     - ConnectionUI.cs
     - PlayerSpawnManager.cs
     - TerrainNetworkSetup.cs

3. **Paste into Unity:**
   - Drag and drop all 7 files into Unity's **Scripts** folder (in Project panel)
   - OR: Copy files, then in Unity right-click Scripts folder → Show in Explorer → Paste

4. **Wait for Compilation:**
   - Unity will automatically compile (check) the scripts
   - You'll see a spinning icon bottom-right
   - If errors appear in Console, don't worry yet - packages might still be loading

**✅ Verification:** In Project > Assets > Scripts, you should see all 7 .cs files with C# icon

---

### 2.4 TextMeshPro (UI Text)

**What it does:** Better text rendering for UI

**Installation (if prompted):**
- When you first use text UI, Unity will prompt: "Import TMP Essentials"
- Click "Import TMP Essentials" button
- Wait a few seconds
- You can skip this for now - we'll do it when creating UI later

---

## ⚙️ Step 3: Configure Project for Meta Quest (Android)

**Why?** Meta Quest runs on Android, so we need to tell Unity to build for Android instead of Windows.

### 3.1 Switch to Android Platform

1. **Open Build Settings:**
   - Top menu → Click `File` → Select `Build Settings`
   - A "Build Settings" window opens

2. **Select Android:**
   - In the "Platform" list (left side), click on **"Android"**
   - It may have a Unity logo icon next to it

3. **Switch Platform:**
   - Bottom-right, click **"Switch Platform"** button
   - **THIS TAKES 10-30 MINUTES!** (First time is slow)
   - You'll see a progress bar
   - Unity is re-importing all assets for Android
   - **Go get coffee/tea - this is a good break!** ☕

4. **Wait for Completion:**
   - When done, the Unity logo will appear next to "Android" in the list
   - This means Android is now the active platform

**✅ Verification:** Android platform shows Unity logo icon next to it in Build Settings

---

### 3.2 Configure Player Settings

**What is this?** These are settings for your app (name, version, icons, etc.)

1. **Open Player Settings:**
   - In the **Build Settings** window (still open from before)
   - Bottom-left, click **"Player Settings..."** button
   - The Inspector panel (right side) will now show "Player" settings

2. **Company & Product Name:**
   - Look for section called **"Company Name"**
   - Type your name or organization name
   - Find **"Product Name"**
   - Enter: `MultiUserTerrain`

3. **Other Settings Section:**
   - In Inspector, scroll down to find **"Other Settings"** section
   - Click the arrow to expand it if it's collapsed

4. **Set Minimum API Level:**
   - Find **"Minimum API Level"** dropdown
   - Select **"Android 10.0 (API level 29)"**
   - (If you see higher, that's fine too)

5. **Set Scripting Backend:**
   - Find **"Scripting Backend"** dropdown
   - Select **"IL2CPP"** (NOT Mono)
   - This is required for Quest

6. **Set Target Architectures:**
   - Find **"Target Architectures"**
   - **Uncheck** "ARMv7" if checked
   - **CHECK** "ARM64" (must be checked!)
   - Only ARM64 should be checked

**✅ Checkpoint:**
- Company Name: Your name
- Product Name: MultiUserTerrain
- Scripting Backend: IL2CPP
- ARM64: Checked

---

### 3.3 Configure XR Settings (VR Support)

**What is this?** Tells Unity this is a VR app for Meta Quest

1. **Open XR Plug-in Management:**
   - Still in Project Settings window
   - On the left sidebar, find and click **"XR Plug-in Management"**
   - If you see "Install XR Plugin Management", click it first and wait

2. **Select Android Tab:**
   - At the top, you'll see tabs: "PC, Mac & Linux Standalone" and "Android"
   - Click the **"Android"** tab (looks like Android robot icon)

3. **Enable Oculus:**
   - You'll see a list of XR providers
   - Find **"Oculus"** (or "Meta Quest")
   - Check the checkbox next to it ✓
   - Unity will install Oculus XR plugin (takes 1-2 minutes)

**✅ Verification:** "Oculus" or "Meta Quest" has a checkmark in Android tab

---

## 🏗️ Step 4: Create Your Scene

**What's a scene?** It's like a level or environment in Unity. We'll build our multi-user terrain world here.

### 4.1 Create New Scene

1. **Create Scene:**
   - Top menu → `File` → `New Scene`
   - A dialog appears with scene templates
   - Select **"Basic (Built-in)"** or **"Empty"**
   - Click "Create" button

2. **Save Scene:**
   - Top menu → `File` → `Save As...`
   - In the save dialog:
     - Navigate to: `Assets` folder (should be default)
     - Create a new folder: Click "New Folder", name it **`Scenes`**
     - Enter that Scenes folder
     - Name your scene: **`MultiUserTerrain`**
     - Click "Save"

**✅ Verification:** In Hierarchy (left panel), you should see your scene name at the top, and maybe "Main Camera" and "Directional Light" objects below it.

---

## Project Folder Structure

```
Assets/
├── Scripts/
│   ├── TerrainInteractionManager.cs  ✓ Created
│   ├── TerrainInputHandler.cs        ✓ Created
│   └── UserActionIndicator.cs        ✓ Created
│
├── Prefabs/
│   ├── NetworkedTerrain.prefab       ⚠ To Create
│   ├── Player.prefab                 ⚠ To Create
│   ├── ManipulationIndicator.prefab  ⚠ To Create
│   └── UserActionIndicator.prefab    ⚠ To Create
│
├── Materials/
│   ├── TerrainMaterial.mat           ⚠ To Create
│   ├── UserColor1.mat                ⚠ To Create
│   ├── UserColor2.mat                ⚠ To Create
│   ├── UserColor3.mat                ⚠ To Create
│   └── UserColor4.mat                ⚠ To Create
│
├── Scenes/
│   └── MultiUserTerrain.unity        ⚠ To Create
│
└── Icons/
    ├── ZoomIcon.png                  ⚠ To Create
    ├── PanIcon.png                   ⚠ To Create
    └── RotateIcon.png                ⚠ To Create
```

---

## 🎯 Step 5: Build the Scene - Add GameObjects

**Now the fun part! We'll add objects to our 3D world.**

### 5.1 Add Network Manager (The "Server Brain")

**What is this?** NetworkManager controls all networking - connections, spawning, etc.

1. **Create Empty GameObject:**
   - In **Hierarchy** panel (left side), right-click in empty space
   - Hover over `Create Empty` → Click it
   - A new "GameObject" appears in the list
   - It's selected and highlighted

2. **Rename to NetworkManager:**
   - With GameObject still selected
   - Either:
     - Press `F2` key, OR
     - Click the name again in Hierarchy
   - Type: `NetworkManager`
   - Press Enter

3. **Add NetworkManager Component:**
   - Make sure "NetworkManager" is selected in Hierarchy
   - Look at **Inspector** panel (right side)
   - At the bottom of Inspector, click **"Add Component"** button
   - A search box appears
   - Type: `NetworkManager`
   - You should see "Network Manager" appear in list (from Unity Netcode package)
   - Click on it to add

4. **Add UnityTransport Component:**
   - Still on NetworkManager object
   - Click **"Add Component"** again
   - Type: `UnityTransport`
   - Click "Unity Transport" to add
   - This handles the actual network communication

5. **Configure NetworkManager:**
   - In Inspector, find the **NetworkManager** component section
   - Look for **"Transport"** field
   - It might be empty or say "None"
   - Click the small circle⭕ icon to the right
   - A selection window opens
   - Double-click on "Unity Transport" in the list
   - OR: Drag the "Unity Transport" component from below up to this field

**✅ Checkpoint:**
- NetworkManager GameObject exists in Hierarchy
- Has NetworkManager component
- Has UnityTransport component
- Transport field is assigned to UnityTransport

**NOTE:** We'll come back to assign Player Prefab and Network Prefabs List later!

---

### 5.2 Add Connection Manager (Networking Controller)

**What is this?** Our custom script that lets users click "Host" or "Join"

1. **Create Another Empty GameObject:**
   - Right-click in Hierarchy → `Create Empty`
   - Rename it to: `ConnectionManager`

2. **Add Our Script:**
   - Select "ConnectionManager" in Hierarchy
   - In Inspector, click **"Add Component"**
   - Type: `NetworkConnectionManager`
   - You should see our script appear (with C# icon)
   - Click it to add

3. **Configure Settings:**
   - In Inspector, find **NetworkConnectionManager** component
   - You'll see fields:
     - **IP Address:** Leave as `127.0.0.1` (for local testing)
     - **Port:** Leave as `7777`
     - **Connection UI:** (leave empty for now)
     - **In Game UI:** (leave empty for now)

**✅ Checkpoint:** ConnectionManager GameObject with NetworkConnectionManager component added

---

### 5.3 Add VR Camera (Your "Eyes" in VR)

**What is this?** The camera that shows what you see in the Quest headset

1. **Delete Default Camera:**
   - In Hierarchy, find "Main Camera"
   - Right-click on it → Select "Delete"
   - Click "OK" if prompted
   - (We're replacing it with VR camera)

2. **Add OVRCameraRig:**
   - In **Project** panel (bottom), use the search box
   - Type: `OVRCameraRig`
   - You should see a prefab (blue cube icon) with that name
   - **Drag** it from Project panel up into **Hierarchy** panel
   - It will appear in your scene

3. **Position the Camera:**
   - Select "OVRCameraRig" in Hierarchy
   - Look at Inspector → Find **Transform** component (top)
   - Set **Position** to:
     - X: `0`
     - Y: `0`
     - Z: `0`
   - (This puts it at world origin)

**✅ Checkpoint:** OVRCameraRig in Hierarchy, Main Camera deleted

---

### 5.4 Create the Terrain Object (What Users Will Interact With)

**What is this?** The 3D model that users will zoom/pan/rotate together

1. **Create a Plane (Flat Surface):**
   - In Hierarchy, right-click
   - Hover over `3D Object` → Click `Plane`
   - A flat square appears in your scene

2. **Rename to Terrain:**
   - Select the "Plane" object
   - Press F2 or click name
   - Type: `Terrain`
   - Press Enter

3. **Position at Waist Height:**
   - Select "Terrain" in Hierarchy
   - In Inspector → Transform component
   - Set **Position:**
     - X: `0`
     - Y: `1` (about waist height in meters)
     - Z: `0`

4. **Scale to Tabletop Size:**
   - Still in Transform component
   - Set **Scale:**
     - X: `0.3`
     - Y: `0.3`
     - Z: `0.3`
   - (This makes it a small tabletop-sized model)

5. **Add NetworkObject Component:**
   - With "Terrain" selected
   - Click "Add Component"
   - Type: `NetworkObject`
   - Add it (from Unity Netcode)
   - This makes it sync across network!

6. **Add TerrainInteractionManager Component:**
   - Still on "Terrain"
   - Click "Add Component"
   - Type: `TerrainInteractionManager`
   - Add our custom script

7. **Configure TerrainInteractionManager:**
   - In Inspector, find **TerrainInteractionManager** component
   - Look for **"Terrain Transform"** field
   - **Drag** the "Terrain" object from Hierarchy into this field
   - (Yes, drag it onto itself - we're telling it which object to move)

   - Set other values:
     - **Zoom Speed:** `0.5`
     - **Pan Speed:** `1.0`
     - **Rotate Speed:** `50`
     - **Zoom Limits:**
       - X (Min): `0.5`
       - Y (Max): `3.0`
     - **Pan Radius:** `5.0`

   - **User Colors** (click arrow to expand):
     - Change **Size** to: `4`
     - Four color slots appear
     - Click each color box and pick colors:
       - Element 0: Red
       - Element 1: Green
       - Element 2: Blue
       - Element 3: Yellow

**✅ Major Checkpoint:**
- Terrain GameObject exists
- Has NetworkObject component
- Has TerrainInteractionManager component
- All settings configured
- Terrain Transform field is assigned

---

### 5.5 Add Player Spawn Manager

**What is this?** Automatically positions players in a circle around the terrain

1. **Create Empty GameObject:**
   - Right-click Hierarchy → Create Empty
   - Rename to: `PlayerSpawnManager`

2. **Add NetworkObject:**
   - Select "PlayerSpawnManager"
   - Add Component → `NetworkObject`

3. **Add PlayerSpawnManager Script:**
   - Click "Add Component"
   - Type: `PlayerSpawnManager`
   - Add our script

4. **Configure:**
   - In Inspector, find **PlayerSpawnManager** component
   - **Terrain Center:** Drag "Terrain" from Hierarchy into this field
   - **Spawn Radius:** `2.0`
   - **Spawn Height:** `1.6`

**✅ Checkpoint:** PlayerSpawnManager setup complete

---

## 📦 Step 6: Create Player Prefab

**What is a prefab?** A reusable template. We'll create one player template that Unity will copy for each person who joins.

### 6.1 Create Player GameObject

1. **Create Empty GameObject:**
   - Right-click in Hierarchy → Create Empty
   - Rename to: `Player`

2. **Add NetworkObject Component:**
   - Select "Player" in Hierarchy
   - Click "Add Component"
   - Type: `NetworkObject`
   - Add it
   - **Important:** This makes each player networked!

3. **Create Child for Input Handling:**
   - Right-click on "Player" in Hierarchy
   - Select `Create Empty`
   - A child object appears under Player (you'll see an arrow/triangle next to Player)
   - Rename this child to: `InputHandler`

4. **Add Scripts to InputHandler:**
   - Select "InputHandler" (the child object)
   - Click "Add Component"
   - Type: `TerrainInputHandler`
   - Add it

   - Click "Add Component" again
   - Type: `TerrainNetworkSetup`
   - Add it
   - (This auto-connects the input handler to the terrain)

**✅ Checkpoint:** Your Hierarchy should look like:
```
Player
  └─ InputHandler
```

---

### 6.2 Convert Player to Prefab

**What we're doing:** Saving Player as a template so Unity can spawn copies for each user

1. **Create Prefabs Folder:**
   - In **Project** panel (bottom), navigate to Assets folder
   - Right-click on "Assets" → Create → Folder
   - Name it: `Prefabs`

2. **Drag Player to Prefabs Folder:**
   - In Hierarchy, click and **DRAG** the "Player" object
   - Drop it into the **Prefabs** folder in Project panel
   - You'll see it turn blue and get a cube icon (this means it's a prefab!)

3. **Delete Player from Scene:**
   - **Important:** Select "Player" in Hierarchy (NOT in Project)
   - Right-click → Delete
   - **Why?** The NetworkManager will spawn players automatically - we don't want one already in the scene!

**✅ Checkpoint:**
- "Player" prefab exists in Project > Assets > Prefabs (blue cube icon)
- NO "Player" in Hierarchy (scene is clean)

---

## 🎨 Step 7: Create Connection UI (Simple Version)

**What is this?** Buttons that let users click "Host" or "Join" to connect.

### 7.1 Create Canvas (UI Container)

1. **Create Canvas:**
   - Right-click in Hierarchy
   - Go to `UI` → Click `Canvas`
   - A "Canvas" appears in Hierarchy
   - You might see a popup about TextMeshPro - click "Import TMP Essentials" if it appears

2. **Configure Canvas Scaler:**
   - Select "Canvas" in Hierarchy
   - In Inspector, find **Canvas Scaler** component
   - Change **UI Scale Mode** dropdown to: `Scale With Screen Size`
   - Set **Reference Resolution:**
     - X: `1920`
     - Y: `1080`

**✅ Checkpoint:** Canvas exists with Canvas Scaler set to Scale With Screen Size

---

### 7.2 Create Host Button

1. **Create Button:**
   - Right-click on "Canvas" in Hierarchy
   - Go to `UI` → Select `Button - TextMeshPro`
   - If TextMeshPro import window appears again, click "Import TMP Essentials"
   - A button appears under Canvas

2. **Rename Button:**
   - Select the new button in Hierarchy
   - Rename it to: `HostButton`

3. **Position the Button:**
   - With HostButton selected
   - In Inspector, find **Rect Transform** component (top)
   - Look for **Anchors** presets (small square icon)
   - Click it, a grid appears
   - Hold **Alt + Shift** and click the **center** box
   - This centers the button

4. **Adjust Position Manually:**
   - In Rect Transform, find **Pos X, Pos Y, Pos Z**
   - Set:
     - Pos X: `0`
     - Pos Y: `50` (above center)
     - Pos Z: `0`

5. **Change Button Text:**
   - In Hierarchy, click the arrow next to "HostButton" to expand it
   - You'll see a child called "Text (TMP)" or similar
   - Select it
   - In Inspector, find **TextMeshPro - Text** component
   - In the **Text Input** box, type: `Start as Host`

**✅ Checkpoint:** HostButton with text "Start as Host" visible in scene

---

### 7.3 Create Client Button (Copy of Host Button)

1. **Duplicate Host Button:**
   - Select "HostButton" in Hierarchy
   - Press `Ctrl+D` (or `Cmd+D` on Mac)
   - A copy appears

2. **Rename:**
   - Rename the copy to: `ClientButton`

3. **Move Below Host Button:**
   - With ClientButton selected
   - In Rect Transform, set:
     - Pos Y: `-10` (below center)

4. **Change Text:**
   - Expand ClientButton
   - Select "Text (TMP)" child
   - Change text to: `Join as Client`

**✅ Checkpoint:** Two buttons visible - "Start as Host" and "Join as Client"

---

### 7.4 Create Disconnect Button

1. **Duplicate Again:**
   - Select "ClientButton"
   - Press `Ctrl+D`
   - Rename to: `DisconnectButton`

2. **Position:**
   - Set Pos Y: `-70` (lower)

3. **Change Text:**
   - Expand DisconnectButton
   - Select Text child
   - Change text to: `Disconnect`

4. **Hide Initially:**
   - Select "DisconnectButton" (parent, not text child)
   - At the very top of Inspector, you'll see the GameObject name and a checkbox
   - **Uncheck** the checkbox next to the name
   - This makes it inactive (it will show when connected)

**✅ Checkpoint:** Three buttons created, DisconnectButton is grayed out/inactive

---

### 7.5 Create Status Text

1. **Create Text:**
   - Right-click on "Canvas"
   - `UI` → `Text - TextMeshPro`
   - Rename to: `StatusText`

2. **Position at Top:**
   - Select StatusText
   - Click Anchors preset icon
   - Hold Alt+Shift and click **top-center** box
   - Set Pos Y: `-50` (slightly below top)

3. **Set Default Text:**
   - In Inspector, find TextMeshPro component
   - Set **Text:** `Ready to connect`
   - Set **Font Size:** `36`
   - Set **Alignment:** Center (horizontal and vertical center buttons)

**✅ Checkpoint:** Status text at top saying "Ready to connect"

---

### 7.6 Connect UI to ConnectionUI Script

**Now we wire up the buttons to actually work!**

1. **Create UI Manager:**
   - Right-click in Hierarchy (NOT under Canvas) → Create Empty
   - Rename to: `UIManager`

2. **Add ConnectionUI Script:**
   - Select UIManager
   - Add Component → `ConnectionUI`
   - You'll see lots of empty fields

3. **Assign Button References:**
   - In Inspector, find **ConnectionUI** component
   - You'll see fields like:

   **Host Button:**
   - Click the small circle ⭕ next to it
   - Find and double-click "HostButton" in the list
   - OR: Drag "HostButton" from Hierarchy directly to this field

   **Client Button:**
   - Drag "ClientButton" to this field

   **Disconnect Button:**
   - Drag "DisconnectButton" to this field

   **Status Text:**
   - Drag "StatusText" to this field

   **Connection Manager:**
   - Drag "ConnectionManager" (from Hierarchy) to this field

4. **Set Default Values:**
   - **Default IP:** `127.0.0.1`
   - **Default Port:** `7777`

**✅ Major Checkpoint:** All UI elements connected to ConnectionUI script!

---

## ⚙️ Step 8: Final Network Configuration

**Now we connect everything together!**

### 8.1 Assign Player Prefab to NetworkManager

1. **Select NetworkManager:**
   - In Hierarchy, click "NetworkManager"

2. **Find Player Prefab Field:**
   - In Inspector, find **NetworkManager** component
   - Look for **Player Prefab** field (might be near top)

3. **Assign Player Prefab:**
   - From **Project** panel, navigate to Assets > Prefabs
   - **Drag** the "Player" prefab to the **Player Prefab** field

**✅ Checkpoint:** Player Prefab field shows "Player" (not "None")

---

### 8.2 Add Network Prefabs to List

**What is this?** We tell NetworkManager which objects can be spawned over network

1. **Still on NetworkManager:**
   - Find **Network Prefabs List** section
   - You'll see "List is Empty"

2. **Add Terrain:**
   - Click the **`+`** button at bottom of list
   - A new slot appears (Element 0)
   - Drag "Terrain" from **Hierarchy** to Element 0

3. **Add Player Prefab:**
   - Click **`+`** again
   - Drag "Player" **prefab** from **Project > Prefabs** to Element 1

4. **Add PlayerSpawnManager:**
   - Click **`+`** again
   - Drag "PlayerSpawnManager" from Hierarchy to Element 2

**✅ Checkpoint:** Network Prefabs List has 3 elements: Terrain, Player, PlayerSpawnManager

---

### 8.3 Connect UI to ConnectionManager

1. **Select ConnectionManager:**
   - In Hierarchy, click "ConnectionManager"

2. **Assign UI References:**
   - In Inspector, find **NetworkConnectionManager** component
   - **Connection UI:** Drag the entire "Canvas" to this field
   - **In Game UI:** Leave empty for now (we don't have in-game UI yet)

**✅ Checkpoint:** ConnectionManager has Canvas assigned

---

## 💾 Step 9: Save Everything!

**VERY IMPORTANT - Save your work!**

1. **Save Scene:**
   - `File` → `Save` (or Ctrl+S)
   - Make sure scene is in Assets/Scenes/

2. **Save Project:**
   - `File` → `Save Project`

**✅ Checkpoint:** Everything saved! (no asterisk * in scene name at top)

---

## 🎮 Step 10: Test in Unity Editor

**Let's see if it works!**

### 10.1 Play in Editor

1. **Click Play Button:**
   - At the top-center of Unity, you'll see ▶ (Play), ⏸ (Pause), ⏭ (Step)
   - Click **▶ Play**
   - The Game view activates

2. **You Should See:**
   - Your UI buttons: "Start as Host", "Join as Client"
   - Status text: "Ready to connect"
   - The terrain plane in the scene

3. **Try Clicking "Start as Host":**
   - Click the "Start as Host" button
   - Watch the **Console** panel (bottom)
   - You should see: "Started as Host" or similar message
   - Buttons should disappear
   - Disconnect button should appear

4. **Click Disconnect:**
   - Click "Disconnect" button
   - Buttons should come back

5. **Stop Play Mode:**
   - Click ▶ button again to stop
   - **IMPORTANT:** Any changes made in Play mode are NOT saved!

**✅ Success Indicators:**
- No red errors in Console
- Buttons work
- Can start as host and disconnect
- Terrain appears in scene

---

## ⚠️ Common Issues & Solutions

### Console Shows Errors:

**Error: "NetworkManager not found"**
- Solution: Make sure NetworkManager GameObject exists in scene
- Make sure it has NetworkManager component

**Error: "OVRCameraRig not found"**
- Solution: Meta XR SDK not properly imported
- Re-import Meta XR SDK from Asset Store

**Error: "Assembly not found" or "Using directive missing"**
- Solution: Unity Netcode package not installed
- Go to Package Manager and install com.unity.netcode.gameobjects

**Error: Script compilation errors on TerrainInputHandler**
- Solution: Meta XR SDK not imported yet
- Import Meta XR SDK first, then import scripts

### Buttons Don't Work:

**Clicking buttons does nothing:**
- Make sure ConnectionUI script is on UIManager
- Make sure all button references are assigned in Inspector
- Check Console for error messages

### Scene Looks Wrong:

**Can't see terrain or buttons:**
- Select "Main Camera" or "OVRCameraRig" in Hierarchy
- Look at Scene view vs Game view (tabs at top of center panel)
- Make sure terrain Position Y is 1 (waist height)

### Network Won't Start:

**"Start as Host" doesn't work:**
- Check Console for errors
- Make sure Player Prefab is assigned in NetworkManager
- Make sure Transport is set to UnityTransport

---

## 🚀 Step 11: Build to Meta Quest (Optional - Advanced)

**Skip this for now if you don't have a Quest. You can test in editor first.**

### 11.1 Enable Developer Mode on Quest

1. Download Meta Quest app on your phone
2. Pair your Quest headset
3. In app settings, enable "Developer Mode"
4. Restart Quest headset

### 11.2 Build Settings

1. **Open Build Settings:**
   - `File` → `Build Settings`

2. **Add Scene:**
   - Click "Add Open Scenes" button
   - Your MultiUserTerrain scene should appear in list

3. **Select Android:**
   - Android should already be selected (Unity logo icon)
   - If not, select it and click "Switch Platform"

4. **Connect Quest:**
   - Plug Quest into PC via USB-C cable
   - Put on headset
   - You'll see "Allow USB Debugging" - click Allow

5. **Build and Run:**
   - In Build Settings, click "Build and Run"
   - Choose a location to save the APK file
   - Name it: `MultiUserTerrain.apk`
   - Click Save
   - **Wait 10-30 minutes** - first build is slow!

6. **App Launches on Quest:**
   - When build completes, app auto-launches on Quest
   - You'll see your UI buttons
   - Put on headset and test!

**✅ Success:** App running on Quest, can see UI and terrain

---

## 📝 Final Checklist

Before considering setup complete, verify:

**Scene Setup:**
- [ ] NetworkManager exists with NetworkManager + UnityTransport components
- [ ] ConnectionManager exists with NetworkConnectionManager script
- [ ] OVRCameraRig exists (replaces Main Camera)
- [ ] Terrain exists with NetworkObject + TerrainInteractionManager
- [ ] PlayerSpawnManager exists
- [ ] Canvas with UI buttons exists
- [ ] UIManager with ConnectionUI script exists

**NetworkManager Configuration:**
- [ ] Transport field = UnityTransport
- [ ] Player Prefab = Player (from Prefabs folder)
- [ ] Network Prefabs List has 3 items (Terrain, Player, PlayerSpawnManager)

**Prefabs:**
- [ ] Player prefab exists in Assets/Prefabs/
- [ ] Player has NetworkObject component
- [ ] Player has InputHandler child with TerrainInputHandler + TerrainNetworkSetup

**UI:**
- [ ] HostButton, ClientButton, DisconnectButton exist
- [ ] StatusText exists
- [ ] All buttons connected to ConnectionUI script
- [ ] ConnectionManager reference assigned

**Testing:**
- [ ] Can click Play and see UI
- [ ] Can click "Start as Host" without errors
- [ ] Terrain visible in scene
- [ ] Console shows no red errors

---

## 🎯 What's Next?

Now that basic setup is complete:

1. **Test Multi-User:**
   - Build to Quest device
   - Run Unity Editor as second instance
   - One starts as Host, other joins as Client
   - See if terrain syncs between both!

2. **Add Input Handling:**
   - Currently buttons work but no hand/controller input yet
   - That requires Quest controllers or hand tracking
   - Will work automatically once on device!

3. **Add Visual Feedback:**
   - Create manipulation indicator (glowing ring around terrain)
   - Add user action indicators (show who's interacting)

4. **Improve UI:**
   - Add IP address input fields
   - Add port input field
   - Add player list
   - Better styling

---

## 🆘 Getting Help

**If you're stuck:**

1. **Check Console:** Bottom panel, look for red errors
2. **Check this guide:** Re-read the step you're on
3. **Check Inspector:** Make sure all fields are assigned (not "None")
4. **Google the error:** Copy exact error message from Console
5. **Unity Manual:** Help → Unity Manual (built-in documentation)

**Useful Resources:**
- Unity Learn: https://learn.unity.com/
- Unity Netcode Docs: https://docs-multiplayer.unity3d.com/netcode/
- Meta XR Docs: https://developer.oculus.com/documentation/unity/

---

## 🎉 Congratulations!

You've successfully set up a multi-user VR terrain interaction system from scratch with ZERO Unity experience!

**What you learned:**
- Unity interface and basics
- Creating GameObjects and Components
- Setting up networking with Unity Netcode
- Configuring Meta Quest VR support
- Building UI in Unity
- Creating and using Prefabs
- Building to Android/Quest

**Next skills to explore:**
- Unity scripting (C#)
- 3D modeling basics
- Advanced networking concepts
- VR interaction patterns

---

**End of Beginner-Friendly Setup Guide** ✅

### NetworkedTerrain.prefab

**Setup:**
1. Select Terrain GameObject in scene
2. Add Component: `NetworkObject` (Unity Netcode)
3. Add Component: `TerrainInteractionManager`
4. Configure TerrainInteractionManager:
   - Terrain Transform: Drag Terrain itself
   - Zoom Speed: 0.5
   - Pan Speed: 1.0
   - Rotate Speed: 50
   - Zoom Limits: Min 0.5, Max 3.0
   - Pan Radius: 5.0
   - User Colors: Create array with 4 different colors
5. Drag to Prefabs folder
6. Add to NetworkManager's Network Prefabs List

### Player.prefab

**Setup:**
1. Create empty GameObject named "Player"
2. Add Component: `NetworkObject`
3. Add child GameObject: "CameraRig" (OVRCameraRig)
4. Add child GameObject: "InputHandler"
5. On InputHandler, add Component: `TerrainInputHandler`
6. On InputHandler, add Component: `UserActionIndicator`
7. Configure TerrainInputHandler:
   - Terrain Manager: (assigned at runtime via script)
8. Drag to Prefabs folder
9. Assign to NetworkManager's Player Prefab slot

### ManipulationIndicator.prefab

**Purpose:** Visual ring/glow around terrain when being manipulated

**Setup:**
1. Create 3D Object > Cylinder (or Torus for ring shape)
2. Name it "ManipulationIndicator"
3. Scale: (1.2, 0.05, 1.2) - slightly larger than terrain
4. Position: (0, 0, 0) - will be child of Terrain
5. Create new Material: "IndicatorMaterial"
   - Shader: Standard or Unlit
   - Rendering Mode: Transparent
   - Albedo: Bright color with alpha ~0.5
   - Emission: Same color
6. Apply material to cylinder
7. Drag to Prefabs folder

**Integration:**
- Assign to TerrainInteractionManager's "Manipulation Indicator" field

### UserActionIndicator.prefab

**Purpose:** Shows what action remote users are performing

**Setup:**
1. Create empty GameObject: "UserActionIndicator"
2. Add Component: `LineRenderer`
   - Positions: 2 (start and end)
   - Width: 0.02
   - Material: Bright colored material
3. Create child GameObject: "IconHolder"
4. Add three UI Canvas children to IconHolder:
   - ZoomIcon (with Image component)
   - PanIcon (with Image component)
   - RotateIcon (with Image component)
5. Add TextMeshPro component for action label
6. Configure UserActionIndicator script:
   - Hand Indicator: LineRenderer
   - Zoom Icon: ZoomIcon GameObject
   - Pan Icon: PanIcon GameObject
   - Rotate Icon: RotateIcon GameObject
   - Action Text: TextMeshPro component
7. Drag to Prefabs folder

---

## Script Integration in Unity

### TerrainInteractionManager

**Attach to:** NetworkedTerrain prefab

**Inspector Settings:**
```
Terrain Transform: [Terrain GameObject]
Zoom Speed: 0.5
Pan Speed: 1.0
Rotate Speed: 50
Zoom Limits: (0.5, 3.0)
Pan Radius: 5.0
Manipulation Indicator: [ManipulationIndicator GameObject]
User Colors:
  - Element 0: Color(1, 0, 0, 1)    // Red
  - Element 1: Color(0, 1, 0, 1)    // Green
  - Element 2: Color(0, 0, 1, 1)    // Blue
  - Element 3: Color(1, 1, 0, 1)    // Yellow
```

### TerrainInputHandler

**Attach to:** Player prefab's InputHandler child

**Inspector Settings:**
```
Terrain Manager: [Assign via script at runtime or manually in scene]
```

**Runtime Assignment Script Example:**
```csharp
void Start()
{
    TerrainInteractionManager terrainMgr =
        FindObjectOfType<TerrainInteractionManager>();
    GetComponent<TerrainInputHandler>().terrainManager = terrainMgr;
}
```

### UserActionIndicator

**Attach to:** Player prefab or spawned per-player

**Inspector Settings:**
```
Hand Indicator: [LineRenderer component]
Zoom Icon: [ZoomIcon GameObject]
Pan Icon: [PanIcon GameObject]
Rotate Icon: [RotateIcon GameObject]
Action Text: [TextMeshPro component]
```

---

## Network Configuration

### NetworkManager Settings

**General:**
- Connection Approval: Disabled (for now)
- Network Tick Rate: 30 Hz
- Max Players: 4-8 (adjust based on needs)

**Transport:**
- Use UnityTransport component
- Protocol Type: UnityTransport (UDP)
- Connection Data: Default

**Player Prefab:**
- Assign Player.prefab

**Network Prefabs List:**
- Add NetworkedTerrain.prefab
- Add Player.prefab
- Add any other networked objects

---

## Testing Setup

### Local Testing (Single Device)
1. Build and deploy to Meta Quest
2. Use Meta XR Simulator for testing in Unity Editor
3. Start as Host in VR
4. Connect simulator as Client

### Multi-Device Testing
1. Build to two Quest devices
2. Ensure both on same network
3. One device: Start as Host
4. Other device(s): Join as Client

---

## Connection Flow

### Host/Server:
```csharp
NetworkManager.Singleton.StartHost();
```

### Client:
```csharp
NetworkManager.Singleton.StartClient();
```

### Connection UI (To Create):
Create simple UI with buttons:
- "Start Host" button → StartHost()
- "Join as Client" button → StartClient()

---

## Build Settings

### Android (Quest) Build Settings:
1. File > Build Settings
2. Switch Platform to Android
3. Add MultiUserTerrain scene
4. Player Settings:
   - Company Name: [Your Name]
   - Product Name: Multi-User Terrain
   - Minimum API Level: Android 10.0 (API 29)
   - Target API Level: Android 12.0 (API 31) or newer
   - IL2CPP Scripting Backend
   - ARM64 Architecture

### XR Settings:
- XR Plug-in Management: Oculus
- OpenXR (optional, but recommended for cross-platform)

---

## Next Steps After Unity Setup

1. **Install Packages:**
   - Unity Netcode for GameObjects
   - Meta XR Core SDK
   - TextMeshPro (if needed)

2. **Create Scene:**
   - Set up NetworkManager
   - Add OVRCameraRig
   - Create terrain object

3. **Create Prefabs:**
   - NetworkedTerrain.prefab
   - Player.prefab
   - ManipulationIndicator.prefab
   - UserActionIndicator.prefab

4. **Assign Scripts:**
   - Attach scripts to appropriate GameObjects
   - Configure inspector values

5. **Test Single Player:**
   - Verify input handling works
   - Test zoom, pan, rotate locally

6. **Test Multiplayer:**
   - Deploy to Quest device
   - Use simulator to test multi-user

7. **Iterate:**
   - Adjust speeds, limits based on feel
   - Improve visual feedback
   - Add polish

---

## Troubleshooting

### Scripts show errors in Unity:
- Ensure all required packages are installed
- Verify using statements at top of scripts
- Check Unity version compatibility

### Network not connecting:
- Ensure NetworkManager is configured
- Check both devices on same network
- Verify Network Prefabs list includes all networked objects

### Hand tracking not working:
- Enable Hand Tracking in Quest settings
- Verify OVRHand components are set up correctly
- Check hand tracking permissions in Android manifest

### Terrain not syncing:
- Verify NetworkObject component on terrain
- Check TerrainInteractionManager is using ServerRPCs
- Ensure ownership is properly assigned

---

## Useful Resources

- Unity Netcode Docs: https://docs-multiplayer.unity3d.com/netcode/current/about/
- Meta XR SDK Docs: https://developer.oculus.com/documentation/unity/
- Unity XR Input: https://docs.unity3d.com/Manual/xr_input.html