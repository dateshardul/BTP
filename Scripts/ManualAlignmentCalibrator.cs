using UnityEngine;
using Unity.Netcode;
using System.Collections.Generic;

/// <summary>
/// Manual spatial alignment using reference point method
/// Fast setup (10 seconds) for same-room multi-user
///
/// How it works:
/// 1. Teacher points at reference object (door, whiteboard, etc.)
/// 2. System records teacher's forward direction as "World North"
/// 3. Students point at same reference object
/// 4. System calculates each student's rotation offset
/// 5. Applies correction so everyone sees correct viewpoint
/// </summary>
public class ManualAlignmentCalibrator : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private SpatialAlignmentManager alignmentManager;
    [SerializeField] private Transform playerHead;  // OVRCameraRig CenterEyeAnchor

    [Header("Calibration Settings")]
    [SerializeField] private float calibrationHoldTime = 1.5f;
    [SerializeField] private LayerMask referenceObjectLayer;  // Objects that can be used as reference

    [Header("Visual Feedback")]
    [SerializeField] private LineRenderer calibrationRay;
    [SerializeField] private GameObject referenceIndicator;  // Shows what you're pointing at
    [SerializeField] private TextMeshPro instructionText;

    // Networked reference direction (from teacher)
    private NetworkVariable<Vector3> worldNorthDirection = new NetworkVariable<Vector3>(
        Vector3.forward,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<bool> calibrationComplete = new NetworkVariable<bool>(
        false,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    // Local calibration state
    private bool isCalibrating = false;
    private float holdTime = 0f;
    private Vector3 pointedDirection = Vector3.forward;

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Auto-find player head if not assigned
        if (playerHead == null)
        {
            GameObject centerEye = GameObject.Find("OVRCameraRig/TrackingSpace/CenterEyeAnchor");
            if (centerEye != null)
            {
                playerHead = centerEye.transform;
            }
        }

        // Subscribe to calibration completion
        calibrationComplete.OnValueChanged += OnCalibrationCompleteChanged;
    }

    public override void OnNetworkDespawn()
    {
        if (calibrationComplete != null) calibrationComplete.OnValueChanged -= OnCalibrationCompleteChanged;
        base.OnNetworkDespawn();
    }

    /// <summary>
    /// Start calibration process
    /// </summary>
    public void StartCalibration()
    {
        isCalibrating = true;
        holdTime = 0f;

        ShowInstructions(IsServer ?
            "TEACHER: Point at reference object (door, whiteboard) and hold Trigger" :
            "STUDENT: Point at same reference object as teacher and hold Trigger");

        if (calibrationRay != null)
        {
            calibrationRay.enabled = true;
        }

        Debug.Log("Calibration started");
    }

    private void Update()
    {
        if (!isCalibrating) return;

        // Check for trigger input (using UnityEngine.Input for simplicity, can use XR input)
        bool triggerHeld = Input.GetKey(KeyCode.Mouse0) || GetTriggerPressed();

        if (triggerHeld)
        {
            holdTime += Time.deltaTime;

            // Update pointed direction
            UpdatePointedDirection();

            // Update visual feedback
            UpdateCalibrationVisuals(holdTime / calibrationHoldTime);

            // Complete calibration after hold time
            if (holdTime >= calibrationHoldTime)
            {
                CompleteCalibration();
            }
        }
        else
        {
            holdTime = 0f;
        }
    }

    /// <summary>
    /// Get trigger state from controller
    /// </summary>
    private bool GetTriggerPressed()
    {
        // Check right controller trigger
        List<UnityEngine.XR.InputDevice> devices = new List<UnityEngine.XR.InputDevice>();
        UnityEngine.XR.InputDevices.GetDevicesAtXRNode(UnityEngine.XR.XRNode.RightHand, devices);

        if (devices.Count > 0)
        {
            float triggerValue;
            if (devices[0].TryGetFeatureValue(UnityEngine.XR.CommonUsages.trigger, out triggerValue))
            {
                return triggerValue > 0.5f;
            }
        }

        return false;
    }

    /// <summary>
    /// Update the direction user is pointing
    /// </summary>
    private void UpdatePointedDirection()
    {
        if (playerHead != null)
        {
            // Use head forward direction (where user is looking)
            pointedDirection = playerHead.forward;

            // Project onto horizontal plane (ignore up/down tilt)
            pointedDirection.y = 0;
            pointedDirection.Normalize();
        }
    }

    /// <summary>
    /// Update visual feedback during calibration
    /// </summary>
    private void UpdateCalibrationVisuals(float progress)
    {
        if (calibrationRay != null && playerHead != null)
        {
            Vector3 rayEnd = playerHead.position + pointedDirection * 5f;

            calibrationRay.SetPosition(0, playerHead.position);
            calibrationRay.SetPosition(1, rayEnd);

            // Color shows progress: White → Green
            Color rayColor = Color.Lerp(Color.white, Color.green, progress);
            calibrationRay.startColor = rayColor;
            calibrationRay.endColor = rayColor;
        }

        if (referenceIndicator != null)
        {
            referenceIndicator.transform.position = playerHead.position + pointedDirection * 3f;
            referenceIndicator.SetActive(true);
        }
    }

    /// <summary>
    /// Complete calibration process
    /// </summary>
    private void CompleteCalibration()
    {
        isCalibrating = false;

        if (IsServer)
        {
            // Teacher: Set world north direction
            SetWorldNorthServerRpc(pointedDirection);
        }
        else
        {
            // Student: Calculate and apply rotation offset
            CalculateAndApplyOffsetServerRpc(pointedDirection);
        }

        // Hide visuals
        if (calibrationRay != null) calibrationRay.enabled = false;
        if (referenceIndicator != null) referenceIndicator.SetActive(false);

        ShowInstructions("Alignment complete!");

        Debug.Log($"Calibration complete. Pointed direction: {pointedDirection}");
    }

    /// <summary>
    /// Teacher sets the world north direction
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    private void SetWorldNorthServerRpc(Vector3 northDirection)
    {
        if (!IsServer) return;

        worldNorthDirection.Value = northDirection.normalized;
        calibrationComplete.Value = true;

        Debug.Log($"World North set to: {northDirection}");
    }

    /// <summary>
    /// Student calculates their rotation offset from world north
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    private void CalculateAndApplyOffsetServerRpc(Vector3 studentPointedDirection)
    {
        // Calculate angle between teacher's north and student's pointed direction
        Vector3 teacherNorth = worldNorthDirection.Value;
        Vector3 studentNorth = studentPointedDirection.normalized;

        // Calculate rotation needed to align student with teacher
        float angle = Vector3.SignedAngle(studentNorth, teacherNorth, Vector3.up);
        Quaternion rotationOffset = Quaternion.Euler(0, angle, 0);

        // Apply to this student's view via alignment manager
        if (alignmentManager != null)
        {
            alignmentManager.SetRotationOffsetServerRpc(rotationOffset);
        }

        Debug.Log($"Student offset calculated: {angle}° rotation");
    }

    /// <summary>
    /// Show instruction text to user
    /// </summary>
    private void ShowInstructions(string message)
    {
        if (instructionText != null)
        {
            instructionText.text = message;
            instructionText.gameObject.SetActive(true);

            // Auto-hide after 3 seconds
            Invoke(nameof(HideInstructions), 3f);
        }

        Debug.Log($"[Calibration] {message}");
    }

    private void HideInstructions()
    {
        if (instructionText != null)
        {
            instructionText.gameObject.SetActive(false);
        }
    }

    private void OnCalibrationCompleteChanged(bool previousValue, bool newValue)
    {
        if (newValue && !IsServer)
        {
            // Server has set world north, student can now calibrate
            ShowInstructions("Teacher has set reference. Now point at same object and hold Trigger");
        }
    }

    // Public properties
    public bool IsCalibrating => isCalibrating;
    public bool IsCalibrated => calibrationComplete.Value;
    public Vector3 WorldNorth => worldNorthDirection.Value;
}

/*
 * USAGE GUIDE - MANUAL ALIGNMENT:
 * ================================
 *
 * TEACHER WORKFLOW:
 * 1. App starts, UI shows: "Point at reference and hold Trigger"
 * 2. Teacher looks at classroom door (or whiteboard, window, etc.)
 * 3. Teacher holds Trigger for 1.5 seconds
 * 4. Ray turns white → green (progress)
 * 5. "Alignment complete!" message
 * 6. Teacher's forward = World North
 *
 * STUDENT WORKFLOW:
 * 1. After teacher completes, UI shows: "Point at same reference"
 * 2. Student looks at same door teacher pointed at
 * 3. Student holds Trigger for 1.5 seconds
 * 4. Ray turns white → green
 * 5. System calculates: "Student is 90° from teacher"
 * 6. Applies -90° rotation correction
 * 7. Now student sees correct viewpoint!
 *
 * EXAMPLE:
 * ========
 *
 * Physical Setup:
 *     [Door/Reference]
 *           ↑
 *           North
 *
 * Teacher → Table ← Student A
 *   (West)         (East, 90° different)
 *
 *         Student B
 *         (South, 180° different)
 *
 * After Calibration:
 * - Teacher sees terrain from west (their view)
 * - Student A sees terrain from east (correct 90° rotated view)
 * - Student B sees terrain from south (correct 180° rotated view)
 * - All perspectives correct!
 *
 * TIPS:
 * =====
 * - Choose a clear, unmistakable reference (door frame, window, whiteboard edge)
 * - Reference should be visible from all positions around table
 * - Use something that won't move (not a chair!)
 * - Recalibrate if someone joins late
 */