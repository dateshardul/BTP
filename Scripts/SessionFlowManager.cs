using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using System.Collections;

/// <summary>
/// Session-based flow manager - Runs EVERY time app is launched
/// Guides users through session setup with clear, intuitive steps
/// Not just first-time - this is the main session startup UI
///
/// Every session workflow:
/// 1. Select role (Teacher/Student) - always ask
/// 2. Connection setup (Host/Join)
/// 3. Spatial alignment (if same room)
/// 4. Session ready - terrain appears
/// 5. In-session UI remains visible with hints and controls
/// </summary>
public class SessionFlowManager : NetworkBehaviour
{
    public enum SessionState
    {
        RoleSelection,      // Choose Teacher or Student
        ConnectionSetup,    // Host or Join
        Connecting,         // Connection in progress
        SpatialSetup,       // Alignment if needed
        SessionActive,      // Active session - terrain visible
        SessionEnded        // Clean disconnect
    }

    [Header("UI Panels")]
    [SerializeField] private GameObject roleSelectionPanel;
    [SerializeField] private GameObject connectionPanel;
    [SerializeField] private GameObject spatialAlignmentPanel;
    [SerializeField] private GameObject activeSessionPanel;  // Stays visible during session
    [SerializeField] private GameObject statusPanel;  // Floating status messages

    [Header("Role Selection UI")]
    [SerializeField] private Button teacherButton;
    [SerializeField] private Button studentButton;
    [SerializeField] private TextMeshProUGUI roleExplanationText;

    [Header("Connection UI")]
    [SerializeField] private Button hostButton;
    [SerializeField] private Button joinButton;
    [SerializeField] private TMP_InputField ipAddressField;
    [SerializeField] private TextMeshProUGUI connectionStatusText;
    [SerializeField] private TextMeshProUGUI connectedUsersText;

    [Header("Alignment UI")]
    [SerializeField] private Button sameRoomButton;
    [SerializeField] private Button remoteButton;
    [SerializeField] private Button startCalibrationButton;
    [SerializeField] private TextMeshProUGUI alignmentInstructionText;
    [SerializeField] private Image calibrationProgressBar;

    [Header("Active Session UI")]
    [SerializeField] private TextMeshProUGUI currentActionText;  // "Teacher: Zooming"
    [SerializeField] private TextMeshProUGUI markerCountText;    // "Markers: 5/50"
    [SerializeField] private Button reanchorButton;
    [SerializeField] private Button clearMarkersButton;
    [SerializeField] private Button endSessionButton;
    [SerializeField] private GameObject controlHintsPanel;  // Always visible, shows current controls

    [Header("Floating Status")]
    [SerializeField] private TextMeshProUGUI floatingStatusText;
    [SerializeField] private float statusDisplayDuration = 3f;

    [Header("References")]
    [SerializeField] private TeacherControlMode teacherControl;
    [SerializeField] private NetworkConnectionManager connectionManager;
    [SerializeField] private SpatialAlignmentManager alignmentManager;

    // State
    private SessionState currentState = SessionState.RoleSelection;
    private TeacherControlMode.UserRole selectedRole;
    private bool isSameRoom = false;

    private void Start()
    {
        // Always start with role selection
        ShowState(SessionState.RoleSelection);
    }

    /// <summary>
    /// Show specific session state UI
    /// </summary>
    private void ShowState(SessionState state)
    {
        currentState = state;

        // Hide all panels
        HideAllPanels();

        // Show relevant panel
        switch (state)
        {
            case SessionState.RoleSelection:
                ShowRoleSelectionPanel();
                break;

            case SessionState.ConnectionSetup:
                ShowConnectionPanel();
                break;

            case SessionState.Connecting:
                ShowConnectingState();
                break;

            case SessionState.SpatialSetup:
                ShowSpatialSetupPanel();
                break;

            case SessionState.SessionActive:
                ShowActiveSessionUI();
                break;

            case SessionState.SessionEnded:
                ShowSessionEndedState();
                break;
        }
    }

    /// <summary>
    /// PANEL 1: Role Selection (ALWAYS shown at start)
    /// </summary>
    private void ShowRoleSelectionPanel()
    {
        if (roleSelectionPanel != null)
        {
            roleSelectionPanel.SetActive(true);
        }

        if (roleExplanationText != null)
        {
            roleExplanationText.text =
                "<size=48><b>Welcome!</b></size>\n\n" +
                "Choose your role for this session:\n\n" +
                "<color=yellow>👨‍🏫 TEACHER</color>\n" +
                "• Control terrain\n• Place markers\n• Lead session\n\n" +
                "<color=cyan>👨‍🎓 STUDENT</color>\n" +
                "• View terrain\n• See teacher actions\n• Learn together";
        }

        // Setup button listeners
        if (teacherButton != null)
        {
            teacherButton.onClick.RemoveAllListeners();
            teacherButton.onClick.AddListener(() =>
            {
                selectedRole = TeacherControlMode.UserRole.Teacher;
                ShowState(SessionState.ConnectionSetup);
                ShowFloatingStatus("Role: Teacher selected", 2f);
            });
        }

        if (studentButton != null)
        {
            studentButton.onClick.RemoveAllListeners();
            studentButton.onClick.AddListener(() =>
            {
                selectedRole = TeacherControlMode.UserRole.Student;
                ShowState(SessionState.ConnectionSetup);
                ShowFloatingStatus("Role: Student selected", 2f);
            });
        }
    }

    /// <summary>
    /// PANEL 2: Connection Setup
    /// </summary>
    private void ShowConnectionPanel()
    {
        if (connectionPanel != null)
        {
            connectionPanel.SetActive(true);
        }

        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            // Teacher hosts
            if (connectionStatusText != null)
            {
                connectionStatusText.text =
                    "<b>Start Your Session</b>\n\n" +
                    "You will host the session.\n" +
                    "Students will join you.\n\n" +
                    "Network: " + GetWiFiName();
            }

            if (hostButton != null)
            {
                hostButton.gameObject.SetActive(true);
                hostButton.onClick.RemoveAllListeners();
                hostButton.onClick.AddListener(() =>
                {
                    if (connectionManager != null)
                    {
                        connectionManager.StartHost();
                    }
                    ShowState(SessionState.Connecting);
                });
            }

            if (joinButton != null) joinButton.gameObject.SetActive(false);
        }
        else
        {
            // Student joins
            if (connectionStatusText != null)
            {
                connectionStatusText.text =
                    "<b>Join Teacher's Session</b>\n\n" +
                    "Enter teacher's session info:\n\n" +
                    "(Ask teacher for IP address)";
            }

            if (joinButton != null)
            {
                joinButton.gameObject.SetActive(true);
                joinButton.onClick.RemoveAllListeners();
                joinButton.onClick.AddListener(() =>
                {
                    if (connectionManager != null)
                    {
                        // Get IP from input field
                        if (ipAddressField != null)
                        {
                            connectionManager.SetIPAddress(ipAddressField.text);
                        }
                        connectionManager.StartClient();
                    }
                    ShowState(SessionState.Connecting);
                });
            }

            if (hostButton != null) hostButton.gameObject.SetActive(false);

            // Show IP input field
            if (ipAddressField != null)
            {
                ipAddressField.gameObject.SetActive(true);
            }
        }
    }

    /// <summary>
    /// STATE 3: Connecting
    /// </summary>
    private void ShowConnectingState()
    {
        ShowFloatingStatus("Connecting...", 0f);  // Indefinite

        StartCoroutine(WaitForConnection());
    }

    private IEnumerator WaitForConnection()
    {
        float timeout = 30f;
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            if (NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient)
            {
                // Connected!
                ShowFloatingStatus("Connected!", 2f);
                yield return new WaitForSeconds(1f);

                // Apply role
                if (teacherControl != null)
                {
                    teacherControl.SetRole(selectedRole);
                }

                // Go to spatial setup
                ShowState(SessionState.SpatialSetup);
                yield break;
            }

            // Update status
            if (connectedUsersText != null && NetworkManager.Singleton != null)
            {
                int userCount = NetworkManager.Singleton.ConnectedClientsList.Count;
                connectedUsersText.text = $"Users connected: {userCount}";
            }

            elapsed += Time.deltaTime;
            yield return null;
        }

        // Timeout
        ShowFloatingStatus("Connection timeout. Please retry.", 5f);
        ShowState(SessionState.ConnectionSetup);
    }

    /// <summary>
    /// PANEL 3: Spatial Alignment
    /// </summary>
    private void ShowSpatialSetupPanel()
    {
        if (spatialAlignmentPanel != null)
        {
            spatialAlignmentPanel.SetActive(true);
        }

        if (alignmentInstructionText != null)
        {
            alignmentInstructionText.text =
                "<b>Setup Type</b>\n\n" +
                "Are all users in the SAME physical room?\n\n" +
                "🏫 <b>Same Room:</b> Quick alignment needed (10s)\n" +
                "🌐 <b>Remote:</b> No alignment needed";
        }

        // Same room button
        if (sameRoomButton != null)
        {
            sameRoomButton.onClick.RemoveAllListeners();
            sameRoomButton.onClick.AddListener(() =>
            {
                isSameRoom = true;
                StartSpatialAlignment();
            });
        }

        // Remote button
        if (remoteButton != null)
        {
            remoteButton.onClick.RemoveAllListeners();
            remoteButton.onClick.AddListener(() =>
            {
                isSameRoom = false;
                ShowFloatingStatus("Remote mode - No alignment needed", 2f);
                ShowState(SessionState.SessionActive);
            });
        }
    }

    /// <summary>
    /// Start spatial alignment calibration
    /// </summary>
    private void StartSpatialAlignment()
    {
        if (alignmentManager != null)
        {
            alignmentManager.StartAlignment();
        }

        // Show instructions based on role
        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            if (alignmentInstructionText != null)
            {
                alignmentInstructionText.text =
                    "<b><color=yellow>Teacher: Set Reference</color></b>\n\n" +
                    "1. Point controller at reference:\n" +
                    "   • Door, whiteboard, window, etc.\n\n" +
                    "2. Hold TRIGGER for 1.5 seconds\n\n" +
                    "Ray will turn: White → Green";
            }
        }
        else
        {
            if (alignmentInstructionText != null)
            {
                alignmentInstructionText.text =
                    "<b><color=cyan>Student: Point at Same Reference</color></b>\n\n" +
                    "1. Ask teacher what they pointed at\n\n" +
                    "2. Point at SAME object\n\n" +
                    "3. Hold TRIGGER for 1.5 seconds\n\n" +
                    "This aligns your view!";
            }
        }

        // Monitor calibration progress
        StartCoroutine(MonitorCalibration());
    }

    private IEnumerator MonitorCalibration()
    {
        var calibrator = FindObjectOfType<ManualAlignmentCalibrator>();

        while (calibrator != null && calibrator.IsCalibrating)
        {
            // Update progress bar
            if (calibrationProgressBar != null)
            {
                // Progress visualization
                // (calibrator would need to expose progress value)
            }

            yield return null;
        }

        // Calibration complete
        ShowFloatingStatus("Alignment complete!", 2f);
        yield return new WaitForSeconds(2f);
        ShowState(SessionState.SessionActive);
    }

    /// <summary>
    /// ACTIVE SESSION: Main UI during teaching/learning
    /// This stays visible throughout the session!
    /// </summary>
    private void ShowActiveSessionUI()
    {
        // Hide setup panels
        HideAllPanels();

        // Show active session panel (stays visible)
        if (activeSessionPanel != null)
        {
            activeSessionPanel.SetActive(true);
        }

        // Setup active session UI based on role
        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            SetupTeacherActiveUI();
        }
        else
        {
            SetupStudentActiveUI();
        }

        // Show terrain now
        ShowFloatingStatus("Session started! Terrain will appear on table.", 3f);
    }

    /// <summary>
    /// Teacher's active session UI (always visible)
    /// </summary>
    private void SetupTeacherActiveUI()
    {
        // Show control hints panel
        if (controlHintsPanel != null)
        {
            controlHintsPanel.SetActive(true);
            // Update with latest control scheme
            var hintsText = controlHintsPanel.GetComponentInChildren<TextMeshProUGUI>();
            if (hintsText != null)
            {
                hintsText.text =
                    "<size=24><b>CONTROLS</b></size>\n" +
                    "Trigger+Move: Zoom\n" +
                    "Trigger+Grip: Pan\n" +
                    "Trigger+Stick: Rotate\n" +
                    "Button A: Marker\n" +
                    "Trigger+B (2s): Reanchor";
            }
        }

        // Setup reanchor button
        if (reanchorButton != null)
        {
            reanchorButton.gameObject.SetActive(true);
            reanchorButton.onClick.RemoveAllListeners();
            reanchorButton.onClick.AddListener(() =>
            {
                ShowFloatingStatus("Point at new surface and hold Trigger+B", 3f);
            });
        }

        // Setup clear markers button
        if (clearMarkersButton != null)
        {
            clearMarkersButton.gameObject.SetActive(true);
            clearMarkersButton.onClick.RemoveAllListeners();
            clearMarkersButton.onClick.AddListener(() =>
            {
                var annotationSystem = FindObjectOfType<AnnotationSystem>();
                if (annotationSystem != null)
                {
                    annotationSystem.RemoveAllMarkersServerRpc();
                    ShowFloatingStatus("All markers cleared", 2f);
                }
            });
        }

        // Setup end session button
        if (endSessionButton != null)
        {
            endSessionButton.gameObject.SetActive(true);
            endSessionButton.onClick.RemoveAllListeners();
            endSessionButton.onClick.AddListener(EndSession);
        }

        // Update marker count continuously
        StartCoroutine(UpdateMarkerCountDisplay());
    }

    /// <summary>
    /// Student's active session UI (minimal, non-intrusive)
    /// </summary>
    private void SetupStudentActiveUI()
    {
        // Show minimal info panel
        if (controlHintsPanel != null)
        {
            controlHintsPanel.SetActive(true);
            var hintsText = controlHintsPanel.GetComponentInChildren<TextMeshProUGUI>();
            if (hintsText != null)
            {
                hintsText.text =
                    "<size=24><b>STUDENT VIEW</b></size>\n" +
                    "Watch colored pointer:\n" +
                    "• Cyan: Pointing\n" +
                    "• Blue: Zooming\n" +
                    "• Green: Panning\n" +
                    "• Yellow: Rotating\n" +
                    "• Red: Marking";
            }
        }

        // Only show end session button for students
        if (endSessionButton != null)
        {
            endSessionButton.gameObject.SetActive(true);
            endSessionButton.onClick.RemoveAllListeners();
            endSessionButton.onClick.AddListener(EndSession);
        }

        // Hide teacher-only buttons
        if (reanchorButton != null) reanchorButton.gameObject.SetActive(false);
        if (clearMarkersButton != null) clearMarkersButton.gameObject.SetActive(false);
    }

    /// <summary>
    /// Update marker count in real-time
    /// </summary>
    private IEnumerator UpdateMarkerCountDisplay()
    {
        var annotationSystem = FindObjectOfType<AnnotationSystem>();

        while (currentState == SessionState.SessionActive)
        {
            if (markerCountText != null && annotationSystem != null)
            {
                int count = annotationSystem.MarkerCount;
                int max = annotationSystem.MaxMarkers;
                markerCountText.text = $"📍 Markers: {count}/{max}";

                // Warning if near limit
                if (count >= max * 0.8f)
                {
                    markerCountText.color = Color.yellow;
                }
                else if (count >= max)
                {
                    markerCountText.color = Color.red;
                }
                else
                {
                    markerCountText.color = Color.white;
                }
            }

            yield return new WaitForSeconds(0.5f);
        }
    }

    /// <summary>
    /// Show floating status message
    /// </summary>
    public void ShowFloatingStatus(string message, float duration)
    {
        if (floatingStatusText != null)
        {
            floatingStatusText.text = message;
            floatingStatusText.gameObject.SetActive(true);

            if (duration > 0)
            {
                StartCoroutine(HideFloatingStatusAfter(duration));
            }
        }

        Debug.Log($"[Session] {message}");
    }

    private IEnumerator HideFloatingStatusAfter(float delay)
    {
        yield return new WaitForSeconds(delay);
        if (floatingStatusText != null)
        {
            floatingStatusText.gameObject.SetActive(false);
        }
    }

    /// <summary>
    /// Update current action text based on what teacher is doing
    /// </summary>
    public void UpdateCurrentAction(string action)
    {
        if (currentActionText != null)
        {
            currentActionText.text = action;
        }
    }

    /// <summary>
    /// End session and return to role selection
    /// </summary>
    private void EndSession()
    {
        // Disconnect
        if (connectionManager != null)
        {
            connectionManager.Disconnect();
        }

        ShowFloatingStatus("Session ended. Disconnecting...", 2f);

        // Return to start after brief delay
        StartCoroutine(ReturnToStart());
    }

    private IEnumerator ReturnToStart()
    {
        yield return new WaitForSeconds(2f);
        ShowState(SessionState.RoleSelection);
    }

    /// <summary>
    /// Hide all UI panels
    /// </summary>
    private void HideAllPanels()
    {
        if (roleSelectionPanel != null) roleSelectionPanel.SetActive(false);
        if (connectionPanel != null) connectionPanel.SetActive(false);
        if (spatialAlignmentPanel != null) spatialAlignmentPanel.SetActive(false);
        if (activeSessionPanel != null) activeSessionPanel.SetActive(false);
    }

    /// <summary>
    /// Get current WiFi network name
    /// </summary>
    private string GetWiFiName()
    {
        // On Quest, this would query Android WiFi
        // For now, return placeholder
        return "Local Network";
    }

    // Public properties and methods
    public SessionState CurrentState => currentState;
    public TeacherControlMode.UserRole SelectedRole => selectedRole;
    public bool IsTeacher => selectedRole == TeacherControlMode.UserRole.Teacher;
    public bool IsSameRoom => isSameRoom;
}

/*
 * PERSISTENT UI DESIGN:
 * =====================
 *
 * ALWAYS VISIBLE DURING SESSION:
 *
 * Teacher View:
 * ┌─────────────────────────────────────────┐
 * │                                         │ ← VR view (transparent background)
 * │              [Terrain on table]         │
 * │                                         │
 * │  ┌──────────────────┐ (bottom-right)   │
 * │  │ CONTROLS         │                   │
 * │  │ Trigger+Move: Zoom│                  │
 * │  │ Button A: Marker  │                  │
 * │  │ Markers: 5/50     │                  │
 * │  │                  │                   │
 * │  │ [Reanchor]       │                   │
 * │  │ [Clear Markers]  │                   │
 * │  │ [End Session]    │                   │
 * │  └──────────────────┘                   │
 * └─────────────────────────────────────────┘
 *
 * Student View:
 * ┌─────────────────────────────────────────┐
 * │                                         │
 * │              [Terrain on table]         │
 * │                                         │
 * │  ┌──────────────────┐ (bottom-right)   │
 * │  │ STUDENT VIEW     │                   │
 * │  │ Pointer Colors:  │                   │
 * │  │ • Cyan: Pointing │                   │
 * │  │ • Blue: Zooming  │                   │
 * │  │ • Green: Panning │                   │
 * │  │                  │                   │
 * │  │ [End Session]    │                   │
 * │  └──────────────────┘                   │
 * └─────────────────────────────────────────┘
 *
 * DESIGN PRINCIPLES:
 * ==================
 * - Semi-transparent panels (don't block view)
 * - Corner placement (non-intrusive)
 * - Can be toggled with menu button
 * - Collapsible to minimize during active use
 * - Expands when looked at (gaze-activated)
 *
 * FLOATING STATUS (Top-center):
 * - Large text, brief messages
 * - "Marker placed!" (2 seconds)
 * - "Terrain reanchored!" (2 seconds)
 * - "Connection lost" (5 seconds + retry button)
 */