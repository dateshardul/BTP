using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Hand gesture-based terrain control for teacher (alternative to controller)
/// Uses Meta Quest 3 hand tracking for natural interactions
/// Gestures: Pinch-zoom, Grab-pan, Twist-rotate, Point-annotate
/// </summary>
public class HandGestureTerrainController : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private TerrainInteractionManager terrainManager;
    [SerializeField] private TeacherControlMode teacherControl;
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private AnnotationSystem annotationSystem;

    [Header("Hand Tracking")]
    [SerializeField] private OVRHand leftHand;
    [SerializeField] private OVRHand rightHand;
    [SerializeField] private OVRSkeleton leftSkeleton;
    [SerializeField] private OVRSkeleton rightSkeleton;

    [Header("Gesture Settings")]
    [SerializeField] private float pinchThreshold = 0.8f;  // How tight pinch must be
    [SerializeField] private float twoHandDistanceThreshold = 0.1f;  // Min distance for two-hand gestures
    [SerializeField] private LayerMask terrainLayer;

    [Header("Control Speeds")]
    [SerializeField] private float zoomSensitivity = 2.0f;
    [SerializeField] private float panSensitivity = 1.0f;
    [SerializeField] private float rotateSensitivity = 100f;

    [Header("Visualization")]
    [SerializeField] private LineRenderer leftPointerRay;
    [SerializeField] private LineRenderer rightPointerRay;
    [SerializeField] private GameObject annotationPreview;  // Shows where marker will be placed

    // Gesture state
    private bool leftPinching = false;
    private bool rightPinching = false;
    private bool isGrabbing = false;
    private bool isTwoHandInteraction = false;

    // Interaction data
    private Vector3 lastLeftHandPos;
    private Vector3 lastRightHandPos;
    private float initialTwoHandDistance;
    private float initialScale;
    private Vector3 lastSingleHandPos;
    private Vector3 annotationTarget;
    private bool annotationTargetValid = false;

    private void Start()
    {
        // Auto-find hand components if not assigned
        if (leftHand == null)
        {
            GameObject leftAnchor = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor");
            if (leftAnchor != null)
            {
                leftHand = leftAnchor.GetComponent<OVRHand>();
                leftSkeleton = leftAnchor.GetComponent<OVRSkeleton>();
            }
        }

        if (rightHand == null)
        {
            GameObject rightAnchor = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor");
            if (rightAnchor != null)
            {
                rightHand = rightAnchor.GetComponent<OVRHand>();
                rightSkeleton = rightAnchor.GetComponent<OVRSkeleton>();
            }
        }

        // Hide annotation preview initially
        if (annotationPreview != null)
        {
            annotationPreview.SetActive(false);
        }
    }

    private void Update()
    {
        // Only allow teacher to control
        if (teacherControl != null && !teacherControl.CanManipulateTerrain)
        {
            DisableVisuals();
            return;
        }

        // Check if hands are tracked
        if (!AreHandsTracked())
        {
            DisableVisuals();
            return;
        }

        // Update gesture detection
        DetectGestures();

        // Update pointer rays
        UpdatePointerRays();

        // Handle interactions based on gestures
        HandleGestureInteractions();
    }

    /// <summary>
    /// Check if hands are currently tracked
    /// </summary>
    private bool AreHandsTracked()
    {
        if (leftHand == null || rightHand == null) return false;

        bool leftTracked = leftHand.IsTracked && leftHand.HandConfidence == OVRHand.TrackingConfidence.High;
        bool rightTracked = rightHand.IsTracked && rightHand.HandConfidence == OVRHand.TrackingConfidence.High;

        return leftTracked || rightTracked;
    }

    /// <summary>
    /// Detect current hand gestures
    /// </summary>
    private void DetectGestures()
    {
        // Check pinch gestures (index finger + thumb together)
        leftPinching = leftHand != null && leftHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        rightPinching = rightHand != null && rightHand.GetFingerIsPinching(OVRHand.HandFinger.Index);

        // Update hand positions
        if (leftHand != null && leftHand.IsTracked)
        {
            lastLeftHandPos = leftHand.transform.position;
        }

        if (rightHand != null && rightHand.IsTracked)
        {
            lastRightHandPos = rightHand.transform.position;
        }

        // Detect two-hand interaction
        bool bothHandsTracked = leftHand != null && rightHand != null &&
                               leftHand.IsTracked && rightHand.IsTracked;

        bool bothPinching = leftPinching && rightPinching;

        float handsDistance = bothHandsTracked ?
            Vector3.Distance(lastLeftHandPos, lastRightHandPos) : 0f;

        isTwoHandInteraction = bothPinching && handsDistance > twoHandDistanceThreshold;
    }

    /// <summary>
    /// Update pointer rays from index fingers
    /// </summary>
    private void UpdatePointerRays()
    {
        // Left hand pointer
        if (leftPointerRay != null && leftHand != null && leftHand.IsTracked)
        {
            Vector3 fingerTip = GetIndexFingerTip(leftSkeleton);
            Vector3 pointerDir = GetPointerDirection(leftHand);

            leftPointerRay.enabled = true;
            leftPointerRay.SetPosition(0, fingerTip);
            leftPointerRay.SetPosition(1, fingerTip + pointerDir * 2f);
            leftPointerRay.startColor = leftPinching ? Color.green : Color.cyan;
            leftPointerRay.endColor = leftPinching ? Color.green : Color.cyan;
        }
        else if (leftPointerRay != null)
        {
            leftPointerRay.enabled = false;
        }

        // Right hand pointer
        if (rightPointerRay != null && rightHand != null && rightHand.IsTracked)
        {
            Vector3 fingerTip = GetIndexFingerTip(rightSkeleton);
            Vector3 pointerDir = GetPointerDirection(rightHand);

            rightPointerRay.enabled = true;
            rightPointerRay.SetPosition(0, fingerTip);
            rightPointerRay.SetPosition(1, fingerTip + pointerDir * 2f);
            rightPointerRay.startColor = rightPinching ? Color.green : Color.cyan;
            rightPointerRay.endColor = rightPinching ? Color.green : Color.cyan;
        }
        else if (rightPointerRay != null)
        {
            rightPointerRay.enabled = false;
        }
    }

    /// <summary>
    /// Handle terrain interactions based on detected gestures
    /// </summary>
    private void HandleGestureInteractions()
    {
        if (isTwoHandInteraction)
        {
            // TWO-HAND GESTURES: Pinch-zoom and twist-rotate
            HandleTwoHandGestures();
        }
        else if (leftPinching || rightPinching)
        {
            // SINGLE-HAND GESTURES: Grab-pan or point-annotate
            HandleSingleHandGestures();
        }
        else
        {
            // No gestures - reset state
            isGrabbing = false;
        }
    }

    /// <summary>
    /// Handle two-hand gestures (pinch-zoom and twist-rotate)
    /// </summary>
    private void HandleTwoHandGestures()
    {
        if (terrainManager == null) return;

        float currentDistance = Vector3.Distance(lastLeftHandPos, lastRightHandPos);

        if (!isGrabbing)
        {
            // Start two-hand interaction
            initialTwoHandDistance = currentDistance;
            initialScale = terrainTransform != null ? terrainTransform.localScale.x : 1f;
            isGrabbing = true;
            terrainManager.StartManipulation();

            Debug.Log("Started two-hand gesture (pinch-zoom/twist-rotate)");
        }

        // PINCH-ZOOM: Distance between hands controls zoom
        float distanceRatio = currentDistance / initialTwoHandDistance;
        float targetScale = initialScale * distanceRatio;
        float currentScale = terrainTransform != null ? terrainTransform.localScale.x : 1f;
        float scaleDelta = (targetScale - currentScale) * zoomSensitivity * Time.deltaTime;

        if (Mathf.Abs(scaleDelta) > 0.001f)
        {
            terrainManager.ZoomTerrain(scaleDelta);
        }

        // TWIST-ROTATE: Angle between hands controls rotation
        Vector3 handsVector = lastRightHandPos - lastLeftHandPos;
        float angle = Mathf.Atan2(handsVector.z, handsVector.x) * Mathf.Rad2Deg;

        // Use center point between hands as rotation pivot
        Vector3 centerPoint = (lastLeftHandPos + lastRightHandPos) * 0.5f;

        // Project center point onto terrain for rotation pivot
        Vector3 projectedCenter = ProjectPointOntoTerrain(centerPoint);

        // Rotate around center point (implementation needed in RotateTerrainAroundPoint)
        // This is simplified - full implementation would track angle changes
    }

    /// <summary>
    /// Handle single-hand gestures (grab-pan or point-annotate)
    /// </summary>
    private void HandleSingleHandGestures()
    {
        Vector3 activeHandPos = rightPinching ? lastRightHandPos : lastLeftHandPos;
        OVRHand activeHand = rightPinching ? rightHand : leftHand;

        if (!isGrabbing)
        {
            // Start single-hand interaction
            lastSingleHandPos = activeHandPos;
            isGrabbing = true;

            // Check if pointing at terrain for annotation
            CheckAnnotationTarget(activeHand);

            if (!annotationTargetValid)
            {
                // Not pointing at terrain - start pan
                terrainManager.StartManipulation();
                Debug.Log("Started single-hand pan gesture");
            }
        }
        else
        {
            if (annotationTargetValid)
            {
                // Maintain annotation preview
                UpdateAnnotationPreview();
            }
            else
            {
                // GRAB-PAN: Move terrain with hand
                Vector3 movement = activeHandPos - lastSingleHandPos;
                Vector3 surfaceMovement = Vector3.ProjectOnPlane(movement, Vector3.up);
                Vector3 panDelta = surfaceMovement * panSensitivity;

                if (panDelta.magnitude > 0.0001f && terrainManager != null)
                {
                    terrainManager.PanTerrain(terrainTransform.position + panDelta);
                }

                lastSingleHandPos = activeHandPos;
            }
        }

        // Release gesture - check if should place annotation
        if (!leftPinching && !rightPinching && isGrabbing)
        {
            if (annotationTargetValid)
            {
                // Place marker at target
                PlaceAnnotationAtTarget();
            }

            isGrabbing = false;
            annotationTargetValid = false;

            if (annotationPreview != null)
            {
                annotationPreview.SetActive(false);
            }

            if (terrainManager != null)
            {
                terrainManager.EndManipulation();
            }
        }
    }

    /// <summary>
    /// Check if hand is pointing at terrain for annotation
    /// </summary>
    private void CheckAnnotationTarget(OVRHand hand)
    {
        if (hand == null || terrainTransform == null) return;

        // Get pointer direction from hand
        Vector3 pointerOrigin = GetIndexFingerTip(hand == leftHand ? leftSkeleton : rightSkeleton);
        Vector3 pointerDirection = GetPointerDirection(hand);

        // Raycast to terrain
        Ray ray = new Ray(pointerOrigin, pointerDirection);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 5f, terrainLayer))
        {
            annotationTarget = hit.point;
            annotationTargetValid = true;

            if (annotationPreview != null)
            {
                annotationPreview.SetActive(true);
                annotationPreview.transform.position = hit.point;
            }
        }
        else
        {
            annotationTargetValid = false;
        }
    }

    /// <summary>
    /// Update annotation preview position
    /// </summary>
    private void UpdateAnnotationPreview()
    {
        if (annotationPreview != null && annotationTargetValid)
        {
            annotationPreview.transform.position = annotationTarget;
        }
    }

    /// <summary>
    /// Place annotation at target point
    /// </summary>
    private void PlaceAnnotationAtTarget()
    {
        if (annotationSystem == null || !annotationTargetValid) return;

        annotationSystem.PlaceMarkerServerRpc(annotationTarget);
        Debug.Log($"Hand gesture: Marker placed at {annotationTarget}");
    }

    /// <summary>
    /// Get index finger tip position
    /// </summary>
    private Vector3 GetIndexFingerTip(OVRSkeleton skeleton)
    {
        if (skeleton == null) return Vector3.zero;

        var bones = skeleton.Bones;
        if (bones == null || bones.Count == 0) return Vector3.zero;

        // Index finger tip is bone ID 19 (or use OVRSkeleton.BoneId.Hand_IndexTip)
        foreach (var bone in bones)
        {
            if (bone.Id == OVRSkeleton.BoneId.Hand_IndexTip)
            {
                return bone.Transform.position;
            }
        }

        return Vector3.zero;
    }

    /// <summary>
    /// Get pointer direction from hand
    /// </summary>
    private Vector3 GetPointerDirection(OVRHand hand)
    {
        if (hand == null) return Vector3.forward;

        // Use hand's forward direction (pointing direction)
        return hand.transform.forward;
    }

    /// <summary>
    /// Project a point onto terrain surface
    /// </summary>
    private Vector3 ProjectPointOntoTerrain(Vector3 point)
    {
        if (terrainTransform == null) return point;

        // Cast ray downward to find terrain
        Ray ray = new Ray(point + Vector3.up * 2f, Vector3.down);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 5f, terrainLayer))
        {
            return hit.point;
        }

        // Fallback: project onto horizontal plane at terrain height
        float terrainY = terrainTransform.position.y;
        return new Vector3(point.x, terrainY, point.z);
    }

    /// <summary>
    /// Disable all visual indicators
    /// </summary>
    private void DisableVisuals()
    {
        if (leftPointerRay != null) leftPointerRay.enabled = false;
        if (rightPointerRay != null) rightPointerRay.enabled = false;
        if (annotationPreview != null) annotationPreview.SetActive(false);
    }

    /// <summary>
    /// Check if hand tracking is better than controller for current user
    /// </summary>
    public bool IsHandTrackingActive()
    {
        return AreHandsTracked() && (leftPinching || rightPinching || isGrabbing);
    }

    // Public getters
    public bool IsTwoHandGesture => isTwoHandInteraction;
    public bool IsLeftPinching => leftPinching;
    public bool IsRightPinching => rightPinching;
    public bool IsGrabbing => isGrabbing;
    public bool AnnotationTargetValid => annotationTargetValid;
    public Vector3 AnnotationTarget => annotationTarget;
}