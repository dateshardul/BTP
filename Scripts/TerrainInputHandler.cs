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