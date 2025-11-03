using UnityEngine;
using UnityEngine.XR;
using Unity.Netcode;
using System.Collections.Generic;

/// <summary>
/// Improved pointer-based terrain control for Meta Quest 3 controller
/// IMPROVED UX Controls:
/// - Button A: Place marker (instant)
/// - Button B: Remove last marker (undo)
/// - Grip + Move Controller: Pan terrain
/// - Index Trigger + Move Controller: Zoom (closer=zoom in, farther=zoom out)
/// - Index Trigger + Thumbstick Left/Right: Rotate around pointer
/// All operations use the point where the controller ray hits the terrain
/// </summary>
public class PointerBasedTerrainController : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private TerrainInteractionManager terrainManager;
    [SerializeField] private TeacherControlMode teacherControl;
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private AnnotationSystem annotationSystem;
    [SerializeField] private SurfaceAnchorManager surfaceAnchor;

    [Header("Controller Settings")]
    [SerializeField] private XRNode controllerNode = XRNode.RightHand;  // Meta Quest 3 right controller
    [SerializeField] private LayerMask terrainLayer;  // Layer for terrain raycast

    [Header("Pointer Visualization")]
    [SerializeField] private LineRenderer pointerLine;
    [SerializeField] private GameObject hitIndicator;  // Visual indicator at hit point
    [SerializeField] private float pointerLength = 10f;

    [Header("Control Speeds")]
    [SerializeField] private float zoomSensitivity = 1.5f;
    [SerializeField] private float panSensitivity = 1.0f;
    [SerializeField] private float rotateSensitivity = 90f;  // Degrees per second
    [SerializeField] private float thumbstickRotateSensitivity = 120f;  // Degrees per second

    [Header("Control Thresholds")]
    [SerializeField] private float thumbstickDeadzone = 0.2f;
    [SerializeField] private float triggerThreshold = 0.1f;
    [SerializeField] private float gripThreshold = 0.1f;

    [Header("Reanchoring")]
    [SerializeField] private float reanchorHoldDuration = 1.5f;  // Hold time to trigger reanchor
    [SerializeField] private LayerMask surfaceLayer;  // Layer for surface detection

    // Current state
    private InputDevice controller;
    private Vector3 lastHitPoint;
    private Vector3 lastControllerPosition;
    private Quaternion lastControllerRotation;
    private float zoomStartDistance;  // For zoom
    private bool lastHitValid = false;

    // Controller input states
    private bool gripPressed = false;
    private bool triggerPressed = false;
    private bool buttonAPressed = false;
    private bool buttonBPressed = false;
    private bool lastButtonAState = false;
    private bool lastButtonBState = false;
    private Vector2 thumbstick;
    private float triggerValue = 0f;
    private float gripValue = 0f;

    // Reanchoring state
    private float reanchorHoldTime = 0f;
    private bool isReanchoring = false;

    private void Start()
    {
        // Get controller device
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(controllerNode, devices);
        if (devices.Count > 0)
        {
            controller = devices[0];
            Debug.Log($"Controller found: {controller.name}");
        }

        // Setup pointer visualization
        if (pointerLine != null)
        {
            pointerLine.startColor = pointerColor;
            pointerLine.endColor = pointerColor;
            pointerLine.startWidth = 0.01f;
            pointerLine.endWidth = 0.005f;
            pointerLine.enabled = false;
        }

        if (hitIndicator != null)
        {
            hitIndicator.SetActive(false);
        }
    }

    private void Update()
    {
        // Only allow teacher to control
        if (teacherControl != null && !teacherControl.CanManipulateTerrain)
        {
            if (pointerLine != null) pointerLine.enabled = false;
            if (hitIndicator != null) hitIndicator.SetActive(false);
            return;
        }

        // Update controller reference if needed
        if (!controller.isValid)
        {
            List<InputDevice> devices = new List<InputDevice>();
            InputDevices.GetDevicesAtXRNode(controllerNode, devices);
            if (devices.Count > 0)
                controller = devices[0];
            return;
        }

        // Read controller inputs
        ReadControllerInput();

        // Update pointer ray
        UpdatePointerRay();

        // Handle mode switching via thumbstick
        HandleModeSwitching();

        // Handle terrain manipulation based on current mode
        HandleTerrainControl();
    }

    /// <summary>
    /// Read all inputs from Meta Quest 3 controller
    /// </summary>
    private void ReadControllerInput()
    {
        // Analog values
        controller.TryGetFeatureValue(CommonUsages.trigger, out triggerValue);
        controller.TryGetFeatureValue(CommonUsages.grip, out gripValue);
        controller.TryGetFeatureValue(CommonUsages.primary2DAxis, out thumbstick);

        // Button states
        triggerPressed = triggerValue > triggerThreshold;
        gripPressed = gripValue > gripThreshold;
        controller.TryGetFeatureValue(CommonUsages.primaryButton, out buttonAPressed);  // A button
        controller.TryGetFeatureValue(CommonUsages.secondaryButton, out buttonBPressed);  // B button
    }

    /// <summary>
    /// Update pointer ray visualization and hit detection
    /// </summary>
    private void UpdatePointerRay()
    {
        // Get controller position and rotation
        Vector3 controllerPos;
        Quaternion controllerRot;

        controller.TryGetFeatureValue(CommonUsages.devicePosition, out controllerPos);
        controller.TryGetFeatureValue(CommonUsages.deviceRotation, out controllerRot);

        // Cast ray from controller
        Ray ray = new Ray(controllerPos, controllerRot * Vector3.forward);
        RaycastHit hit;

        bool hitTerrain = Physics.Raycast(ray, out hit, pointerLength, terrainLayer);
        lastHitValid = hitTerrain && terrainTransform != null;

        if (lastHitValid)
        {
            lastHitPoint = hit.point;

            // Show pointer line
            if (pointerLine != null)
            {
                pointerLine.enabled = true;
                pointerLine.SetPosition(0, controllerPos);
                pointerLine.SetPosition(1, hit.point);

                // Color code based on mode
                pointerLine.startColor = GetModeColor();
                pointerLine.endColor = GetModeColor();
            }

            // Show hit indicator
            if (hitIndicator != null)
            {
                hitIndicator.SetActive(true);
                hitIndicator.transform.position = hit.point;
                hitIndicator.transform.up = hit.normal;  // Align with surface
            }
        }
        else
        {
            // No hit - show ray to max distance
            if (pointerLine != null)
            {
                pointerLine.enabled = true;
                pointerLine.SetPosition(0, controllerPos);
                pointerLine.SetPosition(1, controllerPos + ray.direction * pointerLength);
            }

            if (hitIndicator != null)
            {
                hitIndicator.SetActive(false);
            }
        }

        lastControllerPosition = controllerPos;
    }

    /// <summary>
    /// IMPROVED UX: Handle terrain manipulation with better control scheme
    /// - Button A: Place marker
    /// - Button B: Undo marker
    /// - Grip + Move: Pan
    /// - Trigger + Move: Zoom
    /// - Trigger + Thumbstick: Rotate
    /// </summary>
    private void HandleTerrainControl()
    {
        // REANCHORING - Hold Grip + A + B buttons for 1.5 seconds
        // Point at any surface (table, floor, desk) to reanchor terrain there
        if (gripPressed && buttonAPressed && buttonBPressed)
        {
            reanchorHoldTime += Time.deltaTime;

            if (reanchorHoldTime >= reanchorHoldDuration)
            {
                TriggerReanchor();
                reanchorHoldTime = 0f;
            }
        }
        else
        {
            reanchorHoldTime = 0f;
        }

        if (!lastHitValid && !isReanchoring) return;

        // BUTTON A (alone) - Place Marker (instant action)
        if (buttonAPressed && !buttonBPressed && !gripPressed && !lastButtonAState)
        {
            PlaceMarkerAtPointer();
        }
        lastButtonAState = buttonAPressed && !buttonBPressed && !gripPressed;

        // BUTTON B (alone) - Remove Last Marker (undo)
        if (buttonBPressed && !buttonAPressed && !gripPressed && !lastButtonBState)
        {
            RemoveLastMarker();
        }
        lastButtonBState = buttonBPressed && !buttonAPressed && !gripPressed;

        // GRIP + MOVE - Pan Terrain
        if (gripPressed)
        {
            HandlePan();
        }

        // INDEX TRIGGER + MOVE - Zoom
        // INDEX TRIGGER + THUMBSTICK - Rotate
        if (triggerPressed)
        {
            // Check if using thumbstick for rotation
            if (Mathf.Abs(thumbstick.x) > thumbstickDeadzone)
            {
                HandleRotateWithThumbstick();
            }
            else
            {
                // No thumbstick, use trigger for zoom
                HandleZoom();
            }
        }

        // Update last positions for delta calculations
        Vector3 controllerPos;
        controller.TryGetFeatureValue(CommonUsages.devicePosition, out controllerPos);
        lastControllerPosition = controllerPos;

        Quaternion controllerRot;
        controller.TryGetFeatureValue(CommonUsages.deviceRotation, out controllerRot);
        lastControllerRotation = controllerRot;
    }

    /// <summary>
    /// Place marker at pointer location (Button A)
    /// </summary>
    private void PlaceMarkerAtPointer()
    {
        if (annotationSystem == null || !lastHitValid) return;

        annotationSystem.PlaceMarkerServerRpc(lastHitPoint);
        Debug.Log($"Marker placed at {lastHitPoint}");

        // Visual/haptic feedback
        TriggerHapticFeedback(0.3f, 0.1f);
    }

    /// <summary>
    /// Remove last placed marker (Button B - Undo)
    /// </summary>
    private void RemoveLastMarker()
    {
        if (annotationSystem == null) return;

        annotationSystem.RemoveLastMarkerServerRpc();
        Debug.Log("Last marker removed");

        // Haptic feedback
        TriggerHapticFeedback(0.2f, 0.05f);
    }

    /// <summary>
    /// TRIGGER + MOVE: Zoom terrain by moving controller closer/farther
    /// Zoom is centered on the pointer hit point
    /// </summary>
    private void HandleZoom()
    {
        if (terrainManager == null || !lastHitValid) return;

        // Get current controller position
        Vector3 currentPos;
        controller.TryGetFeatureValue(CommonUsages.devicePosition, out currentPos);

        // Calculate distance change since last frame
        float lastDist = Vector3.Distance(lastControllerPosition, lastHitPoint);
        float currentDist = Vector3.Distance(currentPos, lastHitPoint);
        float distanceDelta = lastDist - currentDist;

        // Convert to zoom (negative = moving closer = zoom in)
        float zoomDelta = distanceDelta * zoomSensitivity;

        if (Mathf.Abs(zoomDelta) > 0.001f)
        {
            terrainManager.ZoomTerrain(zoomDelta);
        }
    }

    /// <summary>
    /// GRIP + MOVE: Pan terrain along surface plane
    /// Terrain follows controller movement
    /// </summary>
    private void HandlePan()
    {
        if (terrainManager == null || terrainTransform == null) return;

        // Get current controller position
        Vector3 currentPos;
        controller.TryGetFeatureValue(CommonUsages.devicePosition, out currentPos);

        // Calculate movement since last frame
        Vector3 movement = currentPos - lastControllerPosition;

        // Project movement onto horizontal plane (table surface)
        Vector3 surfaceMovement = Vector3.ProjectOnPlane(movement, Vector3.up);

        // Apply panning with sensitivity
        Vector3 panDelta = surfaceMovement * panSensitivity;

        if (panDelta.magnitude > 0.0001f)
        {
            terrainManager.PanTerrain(terrainTransform.position + panDelta);
        }
    }

    /// <summary>
    /// TRIGGER + THUMBSTICK: Rotate terrain using thumbstick left/right
    /// Rotation is around vertical axis passing through pointer hit point
    /// </summary>
    private void HandleRotateWithThumbstick()
    {
        if (terrainManager == null || terrainTransform == null || !lastHitValid) return;

        // Use thumbstick horizontal axis for rotation
        float rotationInput = thumbstick.x;  // -1 (left) to +1 (right)

        if (Mathf.Abs(rotationInput) < thumbstickDeadzone) return;

        // Calculate rotation amount (degrees this frame)
        float rotationDelta = rotationInput * thumbstickRotateSensitivity * Time.deltaTime;

        // Rotate terrain around vertical axis through hit point
        RotateTerrainAroundPoint(lastHitPoint, rotationDelta);
    }

    /// <summary>
    /// Rotate terrain around a specific point (pointer location)
    /// </summary>
    private void RotateTerrainAroundPoint(Vector3 pivotPoint, float angleDegrees)
    {
        if (terrainTransform == null) return;

        // Create rotation around vertical (Y) axis
        Quaternion rotation = Quaternion.Euler(0, angleDegrees, 0);

        // Calculate offset from pivot to terrain center
        Vector3 offset = terrainTransform.position - pivotPoint;

        // Rotate the offset vector
        Vector3 rotatedOffset = rotation * offset;

        // Apply new position and rotation
        terrainTransform.position = pivotPoint + rotatedOffset;
        terrainTransform.rotation = rotation * terrainTransform.rotation;

        // Sync via terrain manager
        if (terrainManager != null)
        {
            terrainManager.RotateTerrain(angleDegrees * Time.deltaTime);
        }
    }

    /// <summary>
    /// Trigger haptic feedback on controller
    /// </summary>
    private void TriggerHapticFeedback(float amplitude, float duration)
    {
        if (controller.isValid)
        {
            HapticCapabilities capabilities;
            if (controller.TryGetHapticCapabilities(out capabilities))
            {
                if (capabilities.supportsImpulse)
                {
                    controller.SendHapticImpulse(0, amplitude, duration);
                }
            }
        }
    }

    /// <summary>
    /// Get pointer color based on active controls
    /// </summary>
    private Color GetPointerColor()
    {
        // Reanchor mode takes priority - show progress color
        if (gripPressed && buttonAPressed && buttonBPressed)
        {
            // Fade from white to magenta as hold progresses
            float progress = reanchorHoldTime / reanchorHoldDuration;
            return Color.Lerp(Color.white, Color.magenta, progress);
        }

        // Color based on what's being pressed
        if (buttonAPressed) return Color.red;      // Annotate mode
        if (gripPressed) return Color.green;       // Pan mode
        if (triggerPressed)
        {
            if (Mathf.Abs(thumbstick.x) > thumbstickDeadzone)
                return Color.yellow;  // Rotate mode
            else
                return Color.blue;    // Zoom mode
        }
        return Color.cyan;  // Idle
    }

    /// <summary>
    /// Get color for pointer based on current mode
    /// </summary>
    private Color GetModeColor()
    {
        return GetPointerColor();
    }

    /// <summary>
    /// Reanchor terrain to surface where controller is pointing
    /// Triggered by holding Grip + A + B for 1.5 seconds
    /// </summary>
    private void TriggerReanchor()
    {
        if (surfaceAnchor == null)
        {
            Debug.LogWarning("SurfaceAnchorManager not assigned. Cannot reanchor.");
            return;
        }

        // Get controller ray
        Vector3 controllerPos;
        Quaternion controllerRot;
        controller.TryGetFeatureValue(CommonUsages.devicePosition, out controllerPos);
        controller.TryGetFeatureValue(CommonUsages.deviceRotation, out controllerRot);

        Vector3 rayDirection = controllerRot * Vector3.forward;

        // Raycast to find surface (not just terrain, but any surface)
        Ray ray = new Ray(controllerPos, rayDirection);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, pointerLength, surfaceLayer))
        {
            // Found surface! Reanchor terrain there
            surfaceAnchor.AnchorToPointServerRpc(hit.point, hit.normal);

            Debug.Log($"Terrain reanchored to surface at {hit.point}");

            // Strong haptic feedback
            TriggerHapticFeedback(0.8f, 0.3f);

            isReanchoring = true;
        }
        else
        {
            Debug.LogWarning("No surface detected at controller ray. Point at table/floor/surface.");

            // Error haptic (short pulses)
            TriggerHapticFeedback(0.3f, 0.05f);
        }
    }

    /// <summary>
    /// Public method to trigger reanchor (can be called from UI button)
    /// </summary>
    public void ReanchorToControllerRay()
    {
        TriggerReanchor();
    }

    /// <summary>
    /// Get reanchor progress (0-1) for UI visualization
    /// </summary>
    public float GetReanchorProgress()
    {
        return Mathf.Clamp01(reanchorHoldTime / reanchorHoldDuration);
    }

    // Public getters
    public bool IsGripping => gripPressed;
    public bool IsTriggering => triggerPressed;
    public bool IsZooming => triggerPressed && Mathf.Abs(thumbstick.x) <= thumbstickDeadzone;
    public bool IsRotating => triggerPressed && Mathf.Abs(thumbstick.x) > thumbstickDeadzone;
    public bool IsPanning => gripPressed;
    public bool IsReanchoring => gripPressed && buttonAPressed && buttonBPressed;
    public float ReanchorProgress => GetReanchorProgress();
    public Vector3 LastHitPoint => lastHitPoint;
    public bool PointerHittingTerrain => lastHitValid;
}