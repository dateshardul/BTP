using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Keyboard and Mouse controls for terrain manipulation
/// Useful for: Unity Editor testing, Desktop spectator mode, Development
///
/// CONTROLS:
/// - Left Click + Drag: Pan terrain
/// - Right Click + Drag: Rotate around mouse pointer
/// - Mouse Wheel: Zoom (centered on mouse pointer)
/// - Middle Click: Place marker at mouse pointer
/// - Ctrl + Z: Undo last marker
/// - R: Reset terrain to default
/// - 1-4 Keys: Quick mode indicators (for debugging)
/// </summary>
public class KeyboardMouseTerrainController : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private TerrainInteractionManager terrainManager;
    [SerializeField] private TeacherControlMode teacherControl;
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private AnnotationSystem annotationSystem;
    [SerializeField] private Camera mainCamera;

    [Header("Control Settings")]
    [SerializeField] private bool enableKeyboardMouse = true;
    [SerializeField] private LayerMask terrainLayer;

    [Header("Control Speeds")]
    [SerializeField] private float panSpeed = 0.01f;
    [SerializeField] private float zoomSpeed = 0.5f;
    [SerializeField] private float rotateSpeed = 100f;
    [SerializeField] private float mouseWheelSensitivity = 0.1f;

    [Header("Visualization")]
    [SerializeField] private LineRenderer mouseRay;
    [SerializeField] private GameObject mouseHitIndicator;
    [SerializeField] private bool showMouseRay = true;

    // Mouse state
    private bool isLeftDragging = false;
    private bool isRightDragging = false;
    private Vector3 lastMousePosition;
    private Vector3 mouseHitPoint;
    private bool mouseHitValid = false;

    private void Start()
    {
        // Auto-find camera if not assigned
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
        }

        // Setup mouse ray visualization
        if (mouseRay != null)
        {
            mouseRay.startWidth = 0.005f;
            mouseRay.endWidth = 0.002f;
            mouseRay.enabled = false;
        }

        if (mouseHitIndicator != null)
        {
            mouseHitIndicator.SetActive(false);
        }
    }

    private void Update()
    {
        if (!enableKeyboardMouse) return;

        // Only allow if teacher role (or if no teacher control system)
        if (teacherControl != null && !teacherControl.CanManipulateTerrain)
        {
            DisableVisuals();
            return;
        }

        // Update mouse raycast
        UpdateMouseRaycast();

        // Handle mouse input
        HandleMouseInput();

        // Handle keyboard shortcuts
        HandleKeyboardInput();

        // Update visualizations
        UpdateVisuals();
    }

    /// <summary>
    /// Raycast from mouse position to detect terrain hit
    /// </summary>
    private void UpdateMouseRaycast()
    {
        if (mainCamera == null) return;

        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        mouseHitValid = Physics.Raycast(ray, out hit, 100f, terrainLayer);

        if (mouseHitValid)
        {
            mouseHitPoint = hit.point;
        }
    }

    /// <summary>
    /// Handle mouse button and drag input
    /// </summary>
    private void HandleMouseInput()
    {
        // LEFT CLICK + DRAG - Pan
        if (Input.GetMouseButtonDown(0))
        {
            isLeftDragging = true;
            lastMousePosition = Input.mousePosition;

            if (terrainManager != null)
            {
                terrainManager.StartManipulation();
            }
        }
        else if (Input.GetMouseButtonUp(0))
        {
            isLeftDragging = false;

            if (terrainManager != null)
            {
                terrainManager.EndManipulation();
            }
        }

        if (isLeftDragging)
        {
            HandleMousePan();
        }

        // RIGHT CLICK + DRAG - Rotate around mouse pointer
        if (Input.GetMouseButtonDown(1))
        {
            isRightDragging = true;
            lastMousePosition = Input.mousePosition;

            if (terrainManager != null)
            {
                terrainManager.StartManipulation();
            }
        }
        else if (Input.GetMouseButtonUp(1))
        {
            isRightDragging = false;

            if (terrainManager != null)
            {
                terrainManager.EndManipulation();
            }
        }

        if (isRightDragging && mouseHitValid)
        {
            HandleMouseRotate();
        }

        // MIDDLE CLICK - Place marker
        if (Input.GetMouseButtonDown(2) && mouseHitValid)
        {
            PlaceMarkerAtMouse();
        }

        // MOUSE WHEEL - Zoom centered on mouse pointer
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.01f && mouseHitValid)
        {
            HandleMouseZoom(scroll);
        }
    }

    /// <summary>
    /// Handle keyboard shortcuts
    /// </summary>
    private void HandleKeyboardInput()
    {
        // CTRL + Z - Undo marker
        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyDown(KeyCode.Z))
        {
            UndoMarker();
        }

        // R - Reset terrain
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetTerrain();
        }

        // Number keys for mode debugging (optional)
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Debug.Log("Zoom mode");
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            Debug.Log("Pan mode");
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            Debug.Log("Rotate mode");
        }
        else if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            Debug.Log("Annotate mode");
        }

        // ESC - Stop all manipulation
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            isLeftDragging = false;
            isRightDragging = false;

            if (terrainManager != null)
            {
                terrainManager.EndManipulation();
            }
        }
    }

    /// <summary>
    /// Pan terrain by mouse drag (screen-space movement)
    /// </summary>
    private void HandleMousePan()
    {
        if (terrainManager == null || terrainTransform == null || mainCamera == null) return;

        Vector3 currentMousePos = Input.mousePosition;
        Vector3 mouseDelta = currentMousePos - lastMousePosition;

        // Convert screen space movement to world space
        Vector3 worldDelta = mainCamera.transform.right * mouseDelta.x * panSpeed +
                            mainCamera.transform.up * mouseDelta.y * panSpeed;

        // Project onto horizontal plane (table surface)
        Vector3 surfaceDelta = Vector3.ProjectOnPlane(worldDelta, Vector3.up);

        // Apply panning
        if (surfaceDelta.magnitude > 0.0001f)
        {
            terrainManager.PanTerrain(terrainTransform.position + surfaceDelta);
        }

        lastMousePosition = currentMousePos;
    }

    /// <summary>
    /// Rotate terrain around mouse pointer position
    /// </summary>
    private void HandleMouseRotate()
    {
        if (terrainManager == null || terrainTransform == null) return;

        Vector3 currentMousePos = Input.mousePosition;
        float mouseDeltaX = currentMousePos.x - lastMousePosition.x;

        // Calculate rotation amount (horizontal mouse movement)
        float rotationDelta = mouseDeltaX * rotateSpeed * Time.deltaTime;

        if (Mathf.Abs(rotationDelta) > 0.01f)
        {
            // Rotate around vertical axis through mouse hit point
            RotateTerrainAroundPoint(mouseHitPoint, rotationDelta);
        }

        lastMousePosition = currentMousePos;
    }

    /// <summary>
    /// Zoom terrain centered on mouse pointer position
    /// </summary>
    private void HandleMouseZoom(float scrollDelta)
    {
        if (terrainManager == null) return;

        // Scroll up = zoom in, scroll down = zoom out
        float zoomDelta = scrollDelta * zoomSpeed * mouseWheelSensitivity;

        terrainManager.ZoomTerrain(zoomDelta);

        // Optional: Move terrain toward mouse hit point while zooming
        // This creates a "zoom to cursor" effect like in map applications
        if (terrainTransform != null && mouseHitValid)
        {
            Vector3 toHitPoint = mouseHitPoint - terrainTransform.position;
            Vector3 zoomOffset = toHitPoint * scrollDelta * 0.1f;
            terrainManager.PanTerrain(terrainTransform.position + zoomOffset);
        }
    }

    /// <summary>
    /// Place marker at mouse pointer location
    /// </summary>
    private void PlaceMarkerAtMouse()
    {
        if (annotationSystem == null || !mouseHitValid) return;

        annotationSystem.PlaceMarkerServerRpc(mouseHitPoint);
        Debug.Log($"Marker placed at {mouseHitPoint} (via mouse)");
    }

    /// <summary>
    /// Undo last marker (Ctrl+Z)
    /// </summary>
    private void UndoMarker()
    {
        if (annotationSystem == null) return;

        annotationSystem.RemoveLastMarkerServerRpc();
        Debug.Log("Last marker removed (Ctrl+Z)");
    }

    /// <summary>
    /// Reset terrain to default position/rotation/scale
    /// </summary>
    private void ResetTerrain()
    {
        if (terrainManager == null) return;

        terrainManager.ResetTerrainServerRpc();
        Debug.Log("Terrain reset to default (R key)");
    }

    /// <summary>
    /// Rotate terrain around a specific point (mouse hit location)
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
    /// Update visual indicators
    /// </summary>
    private void UpdateVisuals()
    {
        if (!showMouseRay) return;

        // Update mouse ray
        if (mouseRay != null && mainCamera != null)
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);

            mouseRay.enabled = mouseHitValid;

            if (mouseHitValid)
            {
                mouseRay.SetPosition(0, ray.origin);
                mouseRay.SetPosition(1, mouseHitPoint);

                // Color based on action
                Color rayColor = GetCurrentActionColor();
                mouseRay.startColor = rayColor;
                mouseRay.endColor = rayColor;
            }
        }

        // Update hit indicator
        if (mouseHitIndicator != null)
        {
            mouseHitIndicator.SetActive(mouseHitValid);

            if (mouseHitValid)
            {
                mouseHitIndicator.transform.position = mouseHitPoint;
            }
        }
    }

    /// <summary>
    /// Get color based on current mouse action
    /// </summary>
    private Color GetCurrentActionColor()
    {
        if (Input.GetMouseButton(2)) return Color.red;        // Middle click - Annotate
        if (Input.GetMouseButton(0)) return Color.green;      // Left drag - Pan
        if (Input.GetMouseButton(1)) return Color.yellow;     // Right drag - Rotate
        if (Mathf.Abs(Input.GetAxis("Mouse ScrollWheel")) > 0.01f)
            return Color.blue;  // Scroll - Zoom

        return Color.cyan;  // Idle
    }

    /// <summary>
    /// Disable all visual indicators
    /// </summary>
    private void DisableVisuals()
    {
        if (mouseRay != null) mouseRay.enabled = false;
        if (mouseHitIndicator != null) mouseHitIndicator.SetActive(false);
    }

    /// <summary>
    /// Enable/disable keyboard and mouse controls
    /// </summary>
    public void SetEnabled(bool enabled)
    {
        enableKeyboardMouse = enabled;

        if (!enabled)
        {
            DisableVisuals();
        }
    }

    // Public getters
    public bool IsEnabled => enableKeyboardMouse;
    public bool IsPanning => isLeftDragging;
    public bool IsRotating => isRightDragging;
    public bool IsZooming => Mathf.Abs(Input.GetAxis("Mouse ScrollWheel")) > 0.01f;
    public Vector3 MouseHitPoint => mouseHitPoint;
    public bool MouseHittingTerrain => mouseHitValid;
}

/*
 * KEYBOARD & MOUSE CONTROLS REFERENCE:
 * =====================================
 *
 * MOUSE CONTROLS:
 * - Left Click + Drag     → Pan terrain (green ray)
 * - Right Click + Drag    → Rotate around cursor (yellow ray)
 * - Mouse Wheel Up        → Zoom in (blue ray)
 * - Mouse Wheel Down      → Zoom out (blue ray)
 * - Middle Click          → Place marker (red ray)
 *
 * KEYBOARD SHORTCUTS:
 * - Ctrl + Z              → Undo last marker
 * - R                     → Reset terrain to default
 * - Esc                   → Stop all manipulation
 * - 1, 2, 3, 4            → Debug mode indicators (optional)
 *
 * USAGE SCENARIOS:
 * ================
 *
 * 1. Unity Editor Testing:
 *    - Test terrain manipulation without VR headset
 *    - Quick iteration on control logic
 *    - Debug network synchronization
 *
 * 2. Desktop Spectator Mode:
 *    - Teacher can control from PC while students use VR
 *    - Remote teaching scenarios
 *    - Presentation mode for large screens
 *
 * 3. Development:
 *    - Faster testing cycle (no need to build to Quest)
 *    - Debug marker placement
 *    - Test network features
 *
 * SETUP IN UNITY:
 * ===============
 *
 * Option 1: Add to Teacher Player Prefab (for PC teacher)
 * - Add KeyboardMouseTerrainController component
 * - Assign references (terrain manager, etc.)
 * - Only active when not in VR
 *
 * Option 2: Add to Terrain GameObject (for editor testing)
 * - Add KeyboardMouseTerrainController component
 * - Useful for solo testing in editor
 * - Can be disabled in VR builds
 *
 * COMBINING WITH VR CONTROLS:
 * ===========================
 *
 * The system can auto-detect input method:
 * - If VR headset detected → VR controls active, keyboard/mouse disabled
 * - If no VR headset → Keyboard/mouse controls active
 * - Can be toggled via UI button
 *
 * NETWORK COMPATIBILITY:
 * ======================
 *
 * Works seamlessly with multi-user networking:
 * - Desktop teacher + VR students
 * - VR teacher + Desktop spectators
 * - Mixed groups
 * - All see same terrain state in real-time
 */