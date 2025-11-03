# Multi-User Terrain Interaction System Plan

## Table of Contents
1. [Project Architecture Overview](#project-architecture-overview)
2. [Networking Framework](#networking-framework)
3. [Core Components](#core-components)
4. [Implementation Scripts](#implementation-scripts)
5. [Visual Feedback System](#visual-feedback-system)
6. [Implementation Tips](#implementation-tips)
7. [Next Steps](#next-steps)

## Project Architecture Overview

Based on Meta's Shared Activities in Mixed Reality Motif, this implementation provides a comprehensive multi-user terrain interaction system for collaborative AR/VR experiences.

### Key Features
- **Multi-user synchronization** - All users see the same terrain state
- **Gesture-based interaction** - Hand tracking and controller support
- **Visual feedback** - Clear indicators of what other users are doing
- **Authority management** - Prevents conflicts during simultaneous interactions
- **Cross-platform compatibility** - Works with Meta Quest and other XR devices

## Networking Framework

### Selected: Unity Netcode for GameObjects
- **Pros**: Free, Unity-native, good performance, active development
- **Cons**: More initial setup required compared to Photon
- **Note**: Voice chat removed from requirements (can be added later if needed)

### Setup Requirements
- Meta XR Core SDK (v78+)
- Multiplayer Building Blocks
- Unity Netcode for GameObjects
- Meta XR Platform SDK for invites

## Core Components

### A. Terrain Interaction Manager
Handles zoom, pan, rotate operations with network synchronization.

### B. Network Synchronization
Syncs all terrain transformations across all connected users.

### C. Spawn Manager
Places users around the terrain for optimal viewing angles.

### D. Visual Feedback
Shows what other users are doing with clear visual indicators.

## Implementation Scripts

### TerrainInteractionManager.cs

```csharp
using UnityEngine;
using Fusion;
using Meta.XR.MRUtilityKit;

public class TerrainInteractionManager : NetworkBehaviour
{
    [Header("Terrain References")]
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private float zoomSpeed = 0.5f;
    [SerializeField] private float panSpeed = 1f;
    [SerializeField] private float rotateSpeed = 50f;
    
    [Header("Interaction Limits")]
    [SerializeField] private Vector2 zoomLimits = new Vector2(0.5f, 3f);
    [SerializeField] private float panRadius = 5f;
    
    // Networked properties - synchronized across all clients
    [Networked] public Vector3 NetworkedPosition { get; set; }
    [Networked] public Quaternion NetworkedRotation { get; set; }
    [Networked] public float NetworkedScale { get; set; }
    [Networked] public NetworkBool IsBeingManipulated { get; set; }
    [Networked] public PlayerRef CurrentManipulator { get; set; }
    
    // Local interaction state
    private bool isLocallyGrabbed = false;
    private Vector3 lastHandPosition;
    private Vector2 initialPinchDistance;
    
    // Visual feedback
    [SerializeField] private GameObject manipulationIndicator;
    [SerializeField] private Color[] userColors;
    
    private void Start()
    {
        // Initialize terrain to networked state
        if (Object.HasStateAuthority)
        {
            NetworkedPosition = terrainTransform.position;
            NetworkedRotation = terrainTransform.rotation;
            NetworkedScale = terrainTransform.localScale.x;
        }
    }
    
    public override void FixedUpdateNetwork()
    {
        // Update terrain transform to match networked values
        terrainTransform.position = NetworkedPosition;
        terrainTransform.rotation = NetworkedRotation;
        terrainTransform.localScale = Vector3.one * NetworkedScale;
        
        // Update visual indicators
        UpdateManipulationIndicators();
    }
    
    /// <summary>
    /// Call this when user starts grabbing the terrain
    /// </summary>
    public void StartManipulation()
    {
        if (!IsBeingManipulated)
        {
            isLocallyGrabbed = true;
            RPC_RequestStateAuthority();
        }
    }
    
    /// <summary>
    /// Call this when user releases the terrain
    /// </summary>
    public void EndManipulation()
    {
        isLocallyGrabbed = false;
        RPC_ReleaseStateAuthority();
    }
    
    /// <summary>
    /// Zoom the terrain (pinch gesture or controller trigger)
    /// </summary>
    public void ZoomTerrain(float zoomDelta)
    {
        if (!Object.HasStateAuthority) return;
        
        float newScale = Mathf.Clamp(
            NetworkedScale + zoomDelta * zoomSpeed,
            zoomLimits.x,
            zoomLimits.y
        );
        
        NetworkedScale = newScale;
    }
    
    /// <summary>
    /// Pan the terrain (hand/controller movement)
    /// </summary>
    public void PanTerrain(Vector3 handPosition)
    {
        if (!Object.HasStateAuthority) return;
        
        if (isLocallyGrabbed)
        {
            Vector3 delta = handPosition - lastHandPosition;
            Vector3 newPosition = NetworkedPosition + delta * panSpeed;
            
            // Constrain panning to a radius
            if (newPosition.magnitude <= panRadius)
            {
                NetworkedPosition = newPosition;
            }
        }
        
        lastHandPosition = handPosition;
    }
    
    /// <summary>
    /// Rotate the terrain (twist gesture or controller rotation)
    /// </summary>
    public void RotateTerrain(float rotationDelta)
    {
        if (!Object.HasStateAuthority) return;
        
        Quaternion rotation = Quaternion.Euler(0, rotationDelta * rotateSpeed * Time.deltaTime, 0);
        NetworkedRotation *= rotation;
    }
    
    /// <summary>
    /// Two-handed rotation using hand positions
    /// </summary>
    public void RotateTerrainTwoHanded(Vector3 hand1Pos, Vector3 hand2Pos)
    {
        if (!Object.HasStateAuthority) return;
        
        Vector3 handVector = hand2Pos - hand1Pos;
        float angle = Mathf.Atan2(handVector.z, handVector.x) * Mathf.Rad2Deg;
        NetworkedRotation = Quaternion.Euler(0, angle, 0);
    }
    
    /// <summary>
    /// Pinch-to-zoom using two hands
    /// </summary>
    public void PinchZoom(Vector3 hand1Pos, Vector3 hand2Pos, bool isInitial = false)
    {
        if (!Object.HasStateAuthority) return;
        
        float currentDistance = Vector3.Distance(hand1Pos, hand2Pos);
        
        if (isInitial)
        {
            initialPinchDistance = new Vector2(currentDistance, NetworkedScale);
            return;
        }
        
        float scaleMultiplier = currentDistance / initialPinchDistance.x;
        float newScale = Mathf.Clamp(
            initialPinchDistance.y * scaleMultiplier,
            zoomLimits.x,
            zoomLimits.y
        );
        
        NetworkedScale = newScale;
    }
    
    // RPC to request authority over the terrain
    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_RequestStateAuthority(RpcInfo info = default)
    {
        if (!IsBeingManipulated)
        {
            Object.AssignInputAuthority(info.Source);
            IsBeingManipulated = true;
            CurrentManipulator = info.Source;
        }
    }
    
    // RPC to release authority
    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_ReleaseStateAuthority(RpcInfo info = default)
    {
        if (CurrentManipulator == info.Source)
        {
            Object.RemoveInputAuthority();
            IsBeingManipulated = false;
            CurrentManipulator = PlayerRef.None;
        }
    }
    
    /// <summary>
    /// Visual feedback showing who's manipulating the terrain
    /// </summary>
    private void UpdateManipulationIndicators()
    {
        if (manipulationIndicator != null)
        {
            manipulationIndicator.SetActive(IsBeingManipulated);
            
            if (IsBeingManipulated && CurrentManipulator.IsValid)
            {
                // Set indicator color based on user
                int colorIndex = CurrentManipulator.PlayerId % userColors.Length;
                var renderer = manipulationIndicator.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = userColors[colorIndex];
                }
            }
        }
    }
    
    // Optional: Reset terrain to default state
    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    public void RPC_ResetTerrain(RpcInfo info = default)
    {
        NetworkedPosition = Vector3.zero;
        NetworkedRotation = Quaternion.identity;
        NetworkedScale = 1f;
    }
}
```

### TerrainInputHandler.cs

```csharp
using UnityEngine;
using UnityEngine.XR;
using Meta.XR.MRUtilityKit;

public class TerrainInputHandler : MonoBehaviour
{
    [SerializeField] private TerrainInteractionManager terrainManager;
    
    // XR Input
    private InputDevice leftHand;
    private InputDevice rightHand;
    
    // Hand tracking
    private OVRHand ovrLeftHand;
    private OVRHand ovrRightHand;
    
    // Interaction state
    private bool isGrabbing = false;
    private bool isTwoHandedInteraction = false;
    
    private void Start()
    {
        // Get XR devices
        leftHand = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        rightHand = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        
        // Get OVR Hand components
        ovrLeftHand = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor")
            ?.GetComponent<OVRHand>();
        ovrRightHand = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor")
            ?.GetComponent<OVRHand>();
    }
    
    private void Update()
    {
        // Check for controller or hand input
        HandleControllerInput();
        HandleHandTrackingInput();
    }
    
    private void HandleControllerInput()
    {
        // Grip button for grabbing
        bool leftGrip, rightGrip;
        leftHand.TryGetFeatureValue(CommonUsages.gripButton, out leftGrip);
        rightHand.TryGetFeatureValue(CommonUsages.gripButton, out rightGrip);
        
        bool grabbing = leftGrip || rightGrip;
        
        if (grabbing && !isGrabbing)
        {
            terrainManager.StartManipulation();
            isGrabbing = true;
        }
        else if (!grabbing && isGrabbing)
        {
            terrainManager.EndManipulation();
            isGrabbing = false;
        }
        
        if (isGrabbing)
        {
            // Get controller positions
            Vector3 leftPos, rightPos;
            leftHand.TryGetFeatureValue(CommonUsages.devicePosition, out leftPos);
            rightHand.TryGetFeatureValue(CommonUsages.devicePosition, out rightPos);
            
            // Two-handed manipulation
            if (leftGrip && rightGrip)
            {
                terrainManager.RotateTerrainTwoHanded(leftPos, rightPos);
                terrainManager.PinchZoom(leftPos, rightPos);
            }
            // Single-handed pan
            else if (leftGrip)
            {
                terrainManager.PanTerrain(leftPos);
            }
            else if (rightGrip)
            {
                terrainManager.PanTerrain(rightPos);
            }
            
            // Trigger for zoom
            float leftTrigger, rightTrigger;
            leftHand.TryGetFeatureValue(CommonUsages.trigger, out leftTrigger);
            rightHand.TryGetFeatureValue(CommonUsages.trigger, out rightTrigger);
            
            float zoomDelta = (leftTrigger + rightTrigger - 1f) * Time.deltaTime;
            if (Mathf.Abs(zoomDelta) > 0.01f)
            {
                terrainManager.ZoomTerrain(zoomDelta);
            }
        }
    }
    
    private void HandleHandTrackingInput()
    {
        if (ovrLeftHand == null || ovrRightHand == null) return;
        
        // Check pinch gestures
        bool leftPinching = ovrLeftHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        bool rightPinching = ovrRightHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        
        bool pinching = leftPinching || rightPinching;
        
        if (pinching && !isGrabbing)
        {
            terrainManager.StartManipulation();
            isGrabbing = true;
        }
        else if (!pinching && isGrabbing)
        {
            terrainManager.EndManipulation();
            isGrabbing = false;
        }
        
        if (isGrabbing)
        {
            Vector3 leftPos = ovrLeftHand.transform.position;
            Vector3 rightPos = ovrRightHand.transform.position;
            
            // Two-handed pinch interaction
            if (leftPinching && rightPinching)
            {
                if (!isTwoHandedInteraction)
                {
                    terrainManager.PinchZoom(leftPos, rightPos, isInitial: true);
                    isTwoHandedInteraction = true;
                }
                else
                {
                    terrainManager.PinchZoom(leftPos, rightPos, isInitial: false);
                    terrainManager.RotateTerrainTwoHanded(leftPos, rightPos);
                }
            }
            // Single-handed pan
            else
            {
                isTwoHandedInteraction = false;
                
                if (leftPinching)
                    terrainManager.PanTerrain(leftPos);
                else if (rightPinching)
                    terrainManager.PanTerrain(rightPos);
            }
        }
        else
        {
            isTwoHandedInteraction = false;
        }
    }
}
```

## Visual Feedback System

### UserActionIndicator.cs

```csharp
using UnityEngine;
using Fusion;
using TMPro;

public class UserActionIndicator : NetworkBehaviour
{
    [SerializeField] private LineRenderer handIndicator;
    [SerializeField] private GameObject zoomIcon;
    [SerializeField] private GameObject panIcon;
    [SerializeField] private GameObject rotateIcon;
    [SerializeField] private TextMeshPro actionText;
    
    [Networked] public Vector3 NetworkedHandPosition { get; set; }
    [Networked] public int ActionType { get; set; } // 0=none, 1=zoom, 2=pan, 3=rotate
    
    private void Update()
    {
        // Show indicator at networked hand position
        if (ActionType > 0)
        {
            handIndicator.SetPosition(0, NetworkedHandPosition);
            handIndicator.SetPosition(1, NetworkedHandPosition + Vector3.up * 0.1f);
            
            // Show appropriate icon
            zoomIcon.SetActive(ActionType == 1);
            panIcon.SetActive(ActionType == 2);
            rotateIcon.SetActive(ActionType == 3);
            
            // Update text
            string actionName = ActionType == 1 ? "Zooming" :
                               ActionType == 2 ? "Panning" : "Rotating";
            actionText.text = $"{Object.InputAuthority} is {actionName}";
        }
        
        handIndicator.gameObject.SetActive(ActionType > 0);
    }
}
```

## Implementation Tips

### 1. State Authority Management
- Only the user currently manipulating the terrain has authority to change it
- This prevents conflicts and ensures smooth interactions
- Use RPCs to request and release authority

### 2. Smooth Synchronization
- Use `FixedUpdateNetwork()` for consistent updates across all clients
- Networked properties automatically sync to all connected users
- Consider interpolation for smooth visual updates

### 3. Visual Feedback Best Practices
- Always show who's interacting and what they're doing
- Use colored outlines, icons, or particle effects
- Provide clear visual cues for different interaction types

### 4. Interaction Locking
- Prevent multiple users from manipulating simultaneously
- Or implement proper conflict resolution for collaborative editing
- Consider queuing system for turn-based interactions

### 5. Testing Strategy
- Use Meta XR Simulator to test with multiple users
- Test both hand tracking and controller input
- Verify network synchronization across different devices

## Next Steps

### Phase 1: Setup (Your friend handles most of this)
1. **Install Meta XR Core SDK** (v78+)
2. **Add Multiplayer Building Blocks**
3. **Set up Photon Fusion 2** or Unity Netcode
4. **Configure Meta XR Platform SDK** for invites

### Phase 2: Implementation
1. **Create the terrain interaction scripts** above
2. **Add visual feedback prefabs**
3. **Set up input handling** for both controllers and hand tracking
4. **Test basic functionality** with single user

### Phase 3: Multi-User Testing
1. **Test using Meta XR Simulator** for multiplayer
2. **Verify network synchronization**
3. **Test authority management**
4. **Validate visual feedback system**

### Phase 4: Advanced Features
1. **Add voice chat integration** (if using Photon Fusion 2)
2. **Implement spawn system** around the terrain
3. **Add gesture recognition** for more complex interactions
4. **Create user management system**

## Additional Resources

- [Meta Shared Activities Motif Documentation](https://developer.oculus.com/documentation/unity/unity-mr-multiplayer-building-blocks/)
- [Photon Fusion 2 Documentation](https://doc.photonengine.com/fusion/current)
- [Unity Netcode for GameObjects](https://docs-multiplayer.unity3d.com/netcode/current/about/)
- [Meta XR Platform SDK](https://developer.oculus.com/documentation/unity/unity-xr-platform-sdk/)

---

*This plan provides a comprehensive foundation for implementing multi-user terrain interaction in your Mixed Reality application. The modular design allows for incremental implementation and testing.*
