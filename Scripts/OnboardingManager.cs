using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using System.Collections;

/// <summary>
/// First-time user onboarding and step-by-step guidance system
/// Extremely intuitive UI that guides users through:
/// 1. Role selection (Teacher vs Student)
/// 2. Connection setup (Host vs Join)
/// 3. Spatial alignment (if same room)
/// 4. Control tutorial
/// 5. Ready to teach/learn!
/// </summary>
public class OnboardingManager : NetworkBehaviour
{
    public enum OnboardingStep
    {
        Welcome,              // 0. Welcome screen
        RoleSelection,        // 1. Choose Teacher or Student
        ConnectionSetup,      // 2. Start Host or Join
        WaitingForConnection, // 3. Connecting...
        SpatialAlignment,     // 4. Align spatial coordinates (if same room)
        ControlTutorial,      // 5. Learn controls
        Ready                 // 6. Ready to use!
    }

    [Header("UI References")]
    [SerializeField] private Canvas onboardingCanvas;
    [SerializeField] private TextMeshProUGUI titleText;
    [SerializeField] private TextMeshProUGUI instructionText;
    [SerializeField] private TextMeshProUGUI stepIndicatorText;  // "Step 2 of 6"
    [SerializeField] private Image progressBar;
    [SerializeField] private GameObject buttonContainer;

    [Header("Step-Specific UI")]
    [SerializeField] private Button teacherButton;
    [SerializeField] private Button studentButton;
    [SerializeField] private Button hostButton;
    [SerializeField] private Button joinButton;
    [SerializeField] private Button sameRoomButton;
    [SerializeField] private Button remoteButton;
    [SerializeField] private Button nextButton;
    [SerializeField] private Button skipButton;

    [Header("Visual Guides")]
    [SerializeField] private GameObject handControllerModel;  // 3D model showing controls
    [SerializeField] private GameObject pointerRayExample;    // Animated example of pointer
    [SerializeField] private VideoPlayer tutorialVideo;       // Optional video tutorials

    [Header("Settings")]
    [SerializeField] private bool allowSkip = true;
    [SerializeField] private float autoAdvanceDelay = 0.5f;
    [SerializeField] private bool showOnFirstLaunchOnly = true;

    // State
    private OnboardingStep currentStep = OnboardingStep.Welcome;
    private TeacherControlMode.UserRole selectedRole = TeacherControlMode.UserRole.Student;
    private bool isSameRoom = false;
    private bool hasCompletedOnboarding = false;

    private void Start()
    {
        // Check if user has completed onboarding before
        if (showOnFirstLaunchOnly && PlayerPrefs.GetInt("OnboardingCompleted", 0) == 1)
        {
            hasCompletedOnboarding = true;
            HideOnboarding();
            return;
        }

        // Start onboarding
        ShowStep(OnboardingStep.Welcome);
    }

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Skip onboarding if already connected
        if (NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient)
        {
            hasCompletedOnboarding = true;
            HideOnboarding();
        }
    }

    /// <summary>
    /// Show specific onboarding step
    /// </summary>
    private void ShowStep(OnboardingStep step)
    {
        currentStep = step;

        // Hide all buttons first
        HideAllButtons();

        // Update progress
        UpdateProgress();

        // Show step-specific content
        switch (step)
        {
            case OnboardingStep.Welcome:
                ShowWelcomeStep();
                break;

            case OnboardingStep.RoleSelection:
                ShowRoleSelectionStep();
                break;

            case OnboardingStep.ConnectionSetup:
                ShowConnectionSetupStep();
                break;

            case OnboardingStep.WaitingForConnection:
                ShowWaitingStep();
                break;

            case OnboardingStep.SpatialAlignment:
                ShowSpatialAlignmentStep();
                break;

            case OnboardingStep.ControlTutorial:
                ShowControlTutorialStep();
                break;

            case OnboardingStep.Ready:
                ShowReadyStep();
                break;
        }
    }

    /// <summary>
    /// Step 0: Welcome
    /// </summary>
    private void ShowWelcomeStep()
    {
        SetTitle("Welcome to Multi-User Terrain!");
        SetInstructions(
            "This app lets you explore 3D terrain maps together.\n\n" +
            "One teacher controls the map.\n" +
            "Students view and learn together.\n\n" +
            "Let's get started!"
        );

        ShowButton(nextButton, "Start Setup", () => ShowStep(OnboardingStep.RoleSelection));

        if (allowSkip)
        {
            ShowButton(skipButton, "Skip Tutorial", CompleteOnboarding);
        }
    }

    /// <summary>
    /// Step 1: Role Selection
    /// </summary>
    private void ShowRoleSelectionStep()
    {
        SetTitle("Choose Your Role");
        SetInstructions(
            "Are you the teacher or a student?\n\n" +
            "👨‍🏫 TEACHER:\n" +
            "• Control the terrain map\n" +
            "• Zoom, pan, rotate\n" +
            "• Place markers\n" +
            "• Lead the session\n\n" +
            "👨‍🎓 STUDENT:\n" +
            "• View the terrain\n" +
            "• See teacher's actions\n" +
            "• Learn together"
        );

        ShowButton(teacherButton, "I am the Teacher", () =>
        {
            selectedRole = TeacherControlMode.UserRole.Teacher;
            ShowStep(OnboardingStep.ConnectionSetup);
        });

        ShowButton(studentButton, "I am a Student", () =>
        {
            selectedRole = TeacherControlMode.UserRole.Student;
            ShowStep(OnboardingStep.ConnectionSetup);
        });
    }

    /// <summary>
    /// Step 2: Connection Setup
    /// </summary>
    private void ShowConnectionSetupStep()
    {
        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            // Teacher always hosts
            SetTitle("Start Session");
            SetInstructions(
                "As the teacher, you will host the session.\n\n" +
                "Students will join your session.\n\n" +
                "Make sure you're connected to Wi-Fi."
            );

            ShowButton(hostButton, "Start as Host", () =>
            {
                StartHost();
                ShowStep(OnboardingStep.WaitingForConnection);
            });
        }
        else
        {
            // Student joins
            SetTitle("Join Session");
            SetInstructions(
                "You will join the teacher's session.\n\n" +
                "Make sure you're on the same Wi-Fi network as the teacher.\n\n" +
                "Ask teacher for the session code."
            );

            ShowButton(joinButton, "Join Teacher's Session", () =>
            {
                StartClient();
                ShowStep(OnboardingStep.WaitingForConnection);
            });
        }
    }

    /// <summary>
    /// Step 3: Waiting for connection
    /// </summary>
    private void ShowWaitingStep()
    {
        SetTitle("Connecting...");

        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            SetInstructions(
                "Waiting for students to join...\n\n" +
                "📱 Share this info with students:\n" +
                "• Network: [Your WiFi Name]\n" +
                "• Session Code: [Auto-generated]\n\n" +
                "You can start when ready, or wait for all students."
            );

            ShowButton(nextButton, "Continue (Students will join later)", () =>
            {
                ShowStep(OnboardingStep.SpatialAlignment);
            });
        }
        else
        {
            SetInstructions(
                "Connecting to teacher's session...\n\n" +
                "Please wait..."
            );

            // Auto-advance when connected
            StartCoroutine(WaitForConnection());
        }
    }

    /// <summary>
    /// Step 4: Spatial Alignment (if same room)
    /// </summary>
    private void ShowSpatialAlignmentStep()
    {
        SetTitle("Spatial Setup");
        SetInstructions(
            "Are you and the students in the SAME room?\n" +
            "Or are students joining remotely?\n\n" +
            "🏫 SAME ROOM:\n" +
            "• All around one table\n" +
            "• Need spatial alignment\n" +
            "• 10 second setup\n\n" +
            "🌐 REMOTE:\n" +
            "• Students at different locations\n" +
            "• No alignment needed\n" +
            "• Ready immediately"
        );

        ShowButton(sameRoomButton, "Same Room (align viewpoints)", () =>
        {
            isSameRoom = true;
            StartSpatialAlignment();
        });

        ShowButton(remoteButton, "Remote Students (skip alignment)", () =>
        {
            isSameRoom = false;
            ShowStep(OnboardingStep.ControlTutorial);
        });
    }

    /// <summary>
    /// Start spatial alignment based on role
    /// </summary>
    private void StartSpatialAlignment()
    {
        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            // Teacher calibration
            SetTitle("Teacher: Set Reference");
            SetInstructions(
                "Point your controller at a reference object:\n" +
                "• Classroom door\n" +
                "• Whiteboard edge\n" +
                "• Window frame\n" +
                "• Anything visible from all sides of table\n\n" +
                "Hold TRIGGER for 1.5 seconds\n" +
                "Ray will turn WHITE → GREEN"
            );

            // Trigger manual calibration
            var calibrator = FindObjectOfType<ManualAlignmentCalibrator>();
            if (calibrator != null)
            {
                calibrator.StartCalibration();
            }

            // Auto-advance after calibration
            StartCoroutine(WaitForCalibration(() =>
            {
                ShowStep(OnboardingStep.ControlTutorial);
            }));
        }
        else
        {
            // Student calibration
            SetTitle("Student: Point at Same Reference");
            SetInstructions(
                "Point your controller at the SAME object the teacher pointed at.\n\n" +
                "Hold TRIGGER for 1.5 seconds\n\n" +
                "This ensures you see the correct view of the terrain."
            );

            // Trigger manual calibration
            var calibrator = FindObjectOfType<ManualAlignmentCalibrator>();
            if (calibrator != null)
            {
                calibrator.StartCalibration();
            }

            // Auto-advance after calibration
            StartCoroutine(WaitForCalibration(() =>
            {
                ShowStep(OnboardingStep.ControlTutorial);
            }));
        }
    }

    /// <summary>
    /// Step 5: Control Tutorial
    /// </summary>
    private void ShowControlTutorialStep()
    {
        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            ShowTeacherControlTutorial();
        }
        else
        {
            ShowStudentControlTutorial();
        }
    }

    private void ShowTeacherControlTutorial()
    {
        SetTitle("Teacher Controls");
        SetInstructions(
            "TERRAIN MANIPULATION:\n" +
            "• Trigger + Move → Zoom\n" +
            "• Trigger + Grip + Move → Pan\n" +
            "• Trigger + Thumbstick ← → → Rotate\n\n" +
            "ANNOTATIONS:\n" +
            "• Button A → Place marker\n" +
            "• Button B → Undo marker\n\n" +
            "POINTER:\n" +
            "• Just point → Students see cyan ray\n\n" +
            "Try it now! Point at the terrain and hold Trigger."
        );

        // Show 3D controller model with highlighted buttons
        if (handControllerModel != null)
        {
            handControllerModel.SetActive(true);
        }

        ShowButton(nextButton, "I understand the controls", () =>
        {
            ShowStep(OnboardingStep.Ready);
        });

        if (allowSkip)
        {
            ShowButton(skipButton, "Skip tutorial", () =>
            {
                ShowStep(OnboardingStep.Ready);
            });
        }
    }

    private void ShowStudentControlTutorial()
    {
        SetTitle("Student View");
        SetInstructions(
            "As a student, you can:\n\n" +
            "👁️ WATCH:\n" +
            "• See teacher's pointer (color-coded)\n" +
            "• Watch terrain zoom, pan, rotate\n" +
            "• See markers appear\n\n" +
            "🚶 MOVE:\n" +
            "• Walk around the table\n" +
            "• Get different viewing angles\n" +
            "• Lean in for details\n\n" +
            "You cannot manipulate the terrain.\n" +
            "Focus on learning!"
        );

        ShowButton(nextButton, "I'm ready to learn!", () =>
        {
            ShowStep(OnboardingStep.Ready);
        });
    }

    /// <summary>
    /// Step 6: Ready!
    /// </summary>
    private void ShowReadyStep()
    {
        SetTitle(selectedRole == TeacherControlMode.UserRole.Teacher ?
                "Ready to Teach!" : "Ready to Learn!");

        SetInstructions(
            selectedRole == TeacherControlMode.UserRole.Teacher ?
                "You're all set!\n\n" +
                "The terrain will appear on the table.\n" +
                "Point your controller to show students.\n" +
                "Use controls to manipulate terrain.\n\n" +
                "Good luck with your lesson!" :
                "You're all set!\n\n" +
                "You'll see the terrain on the table.\n" +
                "Watch the teacher's colored pointer.\n" +
                "Walk around for different views.\n\n" +
                "Enjoy learning!"
        );

        ShowButton(nextButton, "Start!", CompleteOnboarding);
    }

    /// <summary>
    /// Complete onboarding and hide UI
    /// </summary>
    private void CompleteOnboarding()
    {
        hasCompletedOnboarding = true;

        // Save completion state
        PlayerPrefs.SetInt("OnboardingCompleted", 1);
        PlayerPrefs.Save();

        // Apply selected role
        var teacherControl = GetComponent<TeacherControlMode>();
        if (teacherControl != null)
        {
            teacherControl.SetRole(selectedRole);
        }

        // Hide onboarding UI
        HideOnboarding();

        // Show brief reminder
        StartCoroutine(ShowBriefReminder());

        Debug.Log("Onboarding completed!");
    }

    /// <summary>
    /// Show brief control reminder after onboarding
    /// </summary>
    private IEnumerator ShowBriefReminder()
    {
        yield return new WaitForSeconds(2f);

        if (selectedRole == TeacherControlMode.UserRole.Teacher)
        {
            ShowFloatingHint("Point controller at terrain to begin", 5f);
        }
        else
        {
            ShowFloatingHint("Watch for teacher's colored pointer", 5f);
        }
    }

    /// <summary>
    /// Show floating hint message
    /// </summary>
    private void ShowFloatingHint(string message, float duration)
    {
        if (instructionText != null)
        {
            instructionText.text = message;
            instructionText.gameObject.SetActive(true);
            StartCoroutine(HideAfterDelay(instructionText.gameObject, duration));
        }
    }

    private IEnumerator HideAfterDelay(GameObject obj, float delay)
    {
        yield return new WaitForSeconds(delay);
        if (obj != null) obj.SetActive(false);
    }

    /// <summary>
    /// Wait for network connection
    /// </summary>
    private IEnumerator WaitForConnection()
    {
        float timeout = 30f;
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            if (NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient)
            {
                // Connected!
                yield return new WaitForSeconds(autoAdvanceDelay);
                ShowStep(OnboardingStep.SpatialAlignment);
                yield break;
            }

            elapsed += Time.deltaTime;
            yield return null;
        }

        // Timeout
        SetInstructions("Connection timed out. Please check network and try again.");
        ShowButton(nextButton, "Retry", () => ShowStep(OnboardingStep.ConnectionSetup));
    }

    /// <summary>
    /// Wait for calibration to complete
    /// </summary>
    private IEnumerator WaitForCalibration(System.Action onComplete)
    {
        var calibrator = FindObjectOfType<ManualAlignmentCalibrator>();

        while (calibrator != null && calibrator.IsCalibrating)
        {
            yield return null;
        }

        yield return new WaitForSeconds(1f);  // Brief pause

        onComplete?.Invoke();
    }

    /// <summary>
    /// Helper: Set title text
    /// </summary>
    private void SetTitle(string title)
    {
        if (titleText != null)
        {
            titleText.text = title;
        }
    }

    /// <summary>
    /// Helper: Set instruction text
    /// </summary>
    private void SetInstructions(string instructions)
    {
        if (instructionText != null)
        {
            instructionText.text = instructions;
        }
    }

    /// <summary>
    /// Helper: Update progress indicator
    /// </summary>
    private void UpdateProgress()
    {
        int stepNumber = (int)currentStep + 1;
        int totalSteps = 7;

        if (stepIndicatorText != null)
        {
            stepIndicatorText.text = $"Step {stepNumber} of {totalSteps}";
        }

        if (progressBar != null)
        {
            progressBar.fillAmount = (float)stepNumber / totalSteps;
        }
    }

    /// <summary>
    /// Helper: Show specific button with text and callback
    /// </summary>
    private void ShowButton(Button button, string text, UnityEngine.Events.UnityAction onClick)
    {
        if (button == null) return;

        button.gameObject.SetActive(true);

        // Set button text
        var buttonText = button.GetComponentInChildren<TextMeshProUGUI>();
        if (buttonText != null)
        {
            buttonText.text = text;
        }

        // Clear existing listeners and add new one
        button.onClick.RemoveAllListeners();
        button.onClick.AddListener(onClick);
    }

    /// <summary>
    /// Helper: Hide all buttons
    /// </summary>
    private void HideAllButtons()
    {
        if (teacherButton != null) teacherButton.gameObject.SetActive(false);
        if (studentButton != null) studentButton.gameObject.SetActive(false);
        if (hostButton != null) hostButton.gameObject.SetActive(false);
        if (joinButton != null) joinButton.gameObject.SetActive(false);
        if (sameRoomButton != null) sameRoomButton.gameObject.SetActive(false);
        if (remoteButton != null) remoteButton.gameObject.SetActive(false);
        if (nextButton != null) nextButton.gameObject.SetActive(false);
        if (skipButton != null) skipButton.gameObject.SetActive(false);
    }

    /// <summary>
    /// Hide entire onboarding UI
    /// </summary>
    private void HideOnboarding()
    {
        if (onboardingCanvas != null)
        {
            onboardingCanvas.gameObject.SetActive(false);
        }

        Debug.Log("Onboarding UI hidden");
    }

    /// <summary>
    /// Show onboarding again (for settings/help)
    /// </summary>
    public void ShowOnboarding()
    {
        if (onboardingCanvas != null)
        {
            onboardingCanvas.gameObject.SetActive(true);
            ShowStep(OnboardingStep.Welcome);
        }
    }

    /// <summary>
    /// Reset onboarding (for testing)
    /// </summary>
    public void ResetOnboarding()
    {
        PlayerPrefs.DeleteKey("OnboardingCompleted");
        PlayerPrefs.Save();
        hasCompletedOnboarding = false;

        ShowOnboarding();
    }

    // Network connection helpers
    private void StartHost()
    {
        if (NetworkManager.Singleton != null)
        {
            NetworkManager.Singleton.StartHost();
        }
    }

    private void StartClient()
    {
        if (NetworkManager.Singleton != null)
        {
            NetworkManager.Singleton.StartClient();
        }
    }

    // Public properties
    public OnboardingStep CurrentStep => currentStep;
    public bool HasCompletedOnboarding => hasCompletedOnboarding;
    public TeacherControlMode.UserRole SelectedRole => selectedRole;
    public bool IsSameRoom => isSameRoom;
}

/*
 * ONBOARDING FLOW DIAGRAM:
 * =========================
 *
 * FIRST-TIME USER PUTS ON HEADSET:
 *
 * Step 1: WELCOME
 * ┌──────────────────────────────┐
 * │ Welcome to Multi-User Terrain│
 * │                              │
 * │ [Start Setup]  [Skip]        │
 * └──────────────────────────────┘
 *         ↓
 * Step 2: ROLE SELECTION
 * ┌──────────────────────────────┐
 * │ Choose Your Role             │
 * │                              │
 * │ [I am the Teacher]           │
 * │ [I am a Student]             │
 * └──────────────────────────────┘
 *    ↓ Teacher        ↓ Student
 *
 * Step 3a: TEACHER                Step 3b: STUDENT
 * ┌──────────────────┐            ┌──────────────────┐
 * │ Start Session    │            │ Join Session     │
 * │ [Start as Host]  │            │ [Join Teacher]   │
 * └──────────────────┘            └──────────────────┘
 *         ↓                              ↓
 *
 * Step 4: CONNECTION
 * ┌──────────────────────────────┐
 * │ Connecting...                │
 * │ [Progress spinner]           │
 * └──────────────────────────────┘
 *         ↓
 *
 * Step 5: SPATIAL ALIGNMENT (if same room)
 * ┌──────────────────────────────┐
 * │ Same room or Remote?         │
 * │                              │
 * │ [Same Room - align]          │
 * │ [Remote - skip]              │
 * └──────────────────────────────┘
 *    ↓ Same Room
 *
 * Step 5a: CALIBRATION
 * ┌──────────────────────────────┐
 * │ Point at reference object    │
 * │ Hold Trigger (1.5s)          │
 * │ ▓▓▓▓▓░░░ [Progress]          │
 * └──────────────────────────────┘
 *         ↓
 *
 * Step 6: CONTROL TUTORIAL
 * ┌──────────────────────────────┐
 * │ Teacher: Controls shown      │
 * │ Student: View tips shown     │
 * │ [3D controller model]        │
 * │ [I understand]               │
 * └──────────────────────────────┘
 *         ↓
 *
 * Step 7: READY!
 * ┌──────────────────────────────┐
 * │ Ready to Teach/Learn!        │
 * │ [Start!]                     │
 * └──────────────────────────────┘
 *         ↓
 *   App starts, UI hides
 *
 * TOTAL TIME:
 * - Teacher: 30-45 seconds
 * - Student: 20-30 seconds
 * - With alignment: +10 seconds
 */