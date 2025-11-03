using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;

/// <summary>
/// Manages teacher-specific UI elements
/// Includes: Reanchor button, control hints, marker count, status display
/// Only visible to users with Teacher role
/// </summary>
public class TeacherUIManager : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private TeacherControlMode teacherControl;
    [SerializeField] private PointerBasedTerrainController terrainController;
    [SerializeField] private AnnotationSystem annotationSystem;

    [Header("UI Elements")]
    [SerializeField] private GameObject teacherUIPanel;  // Parent panel for all teacher UI
    [SerializeField] private Button reanchorButton;
    [SerializeField] private Button clearMarkersButton;
    [SerializeField] private Button resetTerrainButton;
    [SerializeField] private TextMeshProUGUI markerCountText;
    [SerializeField] private TextMeshProUGUI controlHintText;
    [SerializeField] private Image reanchorProgressBar;

    [Header("Settings")]
    [SerializeField] private bool showControlHints = true;
    [SerializeField] private float hintDisplayDuration = 3f;

    private float hintTimer = 0f;
    private string currentHint = "";

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Setup button listeners
        if (reanchorButton != null)
        {
            reanchorButton.onClick.AddListener(OnReanchorButtonClicked);
        }

        if (clearMarkersButton != null)
        {
            clearMarkersButton.onClick.AddListener(OnClearMarkersClicked);
        }

        if (resetTerrainButton != null)
        {
            resetTerrainButton.onClick.AddListener(OnResetTerrainClicked);
        }

        // Hide UI if not teacher
        UpdateUIVisibility();
    }

    public override void OnNetworkDespawn()
    {
        // Remove button listeners
        if (reanchorButton != null)
        {
            reanchorButton.onClick.RemoveListener(OnReanchorButtonClicked);
        }

        if (clearMarkersButton != null)
        {
            clearMarkersButton.onClick.RemoveListener(OnClearMarkersClicked);
        }

        if (resetTerrainButton != null)
        {
            resetTerrainButton.onClick.RemoveListener(OnResetTerrainClicked);
        }

        base.OnNetworkDespawn();
    }

    private void Update()
    {
        if (!IsOwner) return;

        // Update UI visibility based on role
        UpdateUIVisibility();

        // Update marker count
        UpdateMarkerCount();

        // Update control hints
        UpdateControlHints();

        // Update reanchor progress bar
        UpdateReanchorProgress();
    }

    /// <summary>
    /// Show/hide teacher UI based on role
    /// </summary>
    private void UpdateUIVisibility()
    {
        bool isTeacher = teacherControl != null && teacherControl.IsTeacher();

        if (teacherUIPanel != null)
        {
            teacherUIPanel.SetActive(isTeacher && IsOwner);
        }
    }

    /// <summary>
    /// Update marker count display
    /// </summary>
    private void UpdateMarkerCount()
    {
        if (markerCountText != null && annotationSystem != null)
        {
            int count = annotationSystem.MarkerCount;
            int max = annotationSystem.MaxMarkers;
            markerCountText.text = $"Markers: {count}/{max}";
        }
    }

    /// <summary>
    /// Update control hints based on current action
    /// </summary>
    private void UpdateControlHints()
    {
        if (!showControlHints || controlHintText == null || terrainController == null) return;

        string hint = "";

        if (terrainController.IsReanchoring)
        {
            hint = "Hold to reanchor terrain to pointed surface...";
        }
        else if (terrainController.IsPanning)
        {
            hint = "Grip + Move: Panning terrain";
        }
        else if (terrainController.IsZooming)
        {
            hint = "Trigger + Push/Pull: Zooming";
        }
        else if (terrainController.IsRotating)
        {
            hint = "Trigger + Thumbstick: Rotating";
        }

        if (hint != currentHint)
        {
            currentHint = hint;
            hintTimer = hintDisplayDuration;
        }

        if (hintTimer > 0)
        {
            controlHintText.text = currentHint;
            controlHintText.enabled = true;
            hintTimer -= Time.deltaTime;
        }
        else
        {
            controlHintText.enabled = false;
        }
    }

    /// <summary>
    /// Update reanchor progress bar
    /// </summary>
    private void UpdateReanchorProgress()
    {
        if (reanchorProgressBar == null || terrainController == null) return;

        float progress = terrainController.ReanchorProgress;

        reanchorProgressBar.fillAmount = progress;
        reanchorProgressBar.gameObject.SetActive(progress > 0f);
    }

    /// <summary>
    /// Reanchor button clicked
    /// </summary>
    private void OnReanchorButtonClicked()
    {
        if (terrainController == null)
        {
            Debug.LogWarning("Terrain controller not assigned");
            return;
        }

        // Trigger reanchor at controller ray
        terrainController.ReanchorToControllerRay();

        // Show hint
        ShowHint("Point controller at surface and press Trigger to reanchor");
    }

    /// <summary>
    /// Clear all markers button clicked
    /// </summary>
    private void OnClearMarkersClicked()
    {
        if (annotationSystem == null)
        {
            Debug.LogWarning("Annotation system not assigned");
            return;
        }

        annotationSystem.RemoveAllMarkersServerRpc();
        ShowHint("All markers cleared");
    }

    /// <summary>
    /// Reset terrain button clicked
    /// </summary>
    private void OnResetTerrainClicked()
    {
        if (terrainController == null || terrainController.IsOwner == false)
        {
            Debug.LogWarning("Cannot reset terrain - not owner");
            return;
        }

        // Get terrain interaction manager and reset
        var terrainManager = FindObjectOfType<TerrainInteractionManager>();
        if (terrainManager != null)
        {
            terrainManager.ResetTerrainServerRpc();
            ShowHint("Terrain reset to default position");
        }
    }

    /// <summary>
    /// Show a hint message
    /// </summary>
    public void ShowHint(string hint)
    {
        currentHint = hint;
        hintTimer = hintDisplayDuration;

        if (controlHintText != null)
        {
            controlHintText.text = hint;
            controlHintText.enabled = true;
        }
    }

    /// <summary>
    /// Public method to trigger reanchor (can be called from other scripts)
    /// </summary>
    public void TriggerReanchor()
    {
        OnReanchorButtonClicked();
    }
}

/*
 * TEACHER UI SETUP GUIDE:
 * =======================
 *
 * Create a Canvas for Teacher UI:
 *
 * 1. Create UI Canvas:
 *    - Right-click Hierarchy → UI → Canvas
 *    - Rename to "TeacherUI"
 *    - Canvas Scaler → UI Scale Mode: Scale With Screen Size
 *
 * 2. Create Panel (Background):
 *    - Right-click TeacherUI → UI → Panel
 *    - Rename to "TeacherControlPanel"
 *    - Position in bottom-right corner
 *    - Size: 300 x 200
 *
 * 3. Add Buttons:
 *    - Create "ReanchorButton": Text = "Reanchor Terrain"
 *    - Create "ClearMarkersButton": Text = "Clear All Markers"
 *    - Create "ResetTerrainButton": Text = "Reset Terrain"
 *    - Stack vertically
 *
 * 4. Add Text Displays:
 *    - MarkerCountText: Shows "Markers: 5/50"
 *    - ControlHintText: Shows current action hint
 *
 * 5. Add Progress Bar (for reanchor hold):
 *    - UI → Slider (remove handle)
 *    - Use Fill image as progress bar
 *    - Color: White → Magenta
 *
 * 6. Add TeacherUIManager Component:
 *    - Add to TeacherUI GameObject
 *    - Assign all UI references
 *    - Assign TeacherControlMode reference
 *    - Assign PointerBasedTerrainController reference
 *    - Assign AnnotationSystem reference
 *
 * REANCHOR METHODS:
 * =================
 *
 * Method 1: Controller Combo (Always Available)
 * - Hold Grip + A + B for 1.5 seconds
 * - Point at surface
 * - Pointer fades white → magenta
 * - Release when magenta
 * - Terrain moves to pointed surface
 *
 * Method 2: UI Button (Optional)
 * - Click "Reanchor Terrain" button
 * - Point controller at surface
 * - Press Trigger
 * - Terrain moves there
 *
 * INITIAL PLACEMENT:
 * ==================
 *
 * Option A: Auto-anchor (Recommended - Fast)
 * - autoAnchorOnStart = true
 * - askTeacherFirst = false
 * - Terrain appears on nearest table immediately
 * - Teacher can reanchor if needed
 *
 * Option B: Ask Teacher First
 * - autoAnchorOnStart = false
 * - askTeacherFirst = true
 * - Show UI: "Point at table and press Trigger to place terrain"
 * - Teacher selects location
 * - More control, but slower start
 */