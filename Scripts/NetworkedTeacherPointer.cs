using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Networked pointer visualization for teacher
/// Shows all students where the teacher is pointing in real-time
/// Displays as a colored ray with hit indicator
/// </summary>
public class NetworkedTeacherPointer : NetworkBehaviour
{
    [Header("Pointer Settings")]
    [SerializeField] private bool alwaysVisible = true;  // Show pointer even when not manipulating
    [SerializeField] private float pointerWidth = 0.015f;
    [SerializeField] private float pointerLength = 10f;

    [Header("Visual Components")]
    [SerializeField] private LineRenderer pointerRay;
    [SerializeField] private GameObject hitIndicator;  // Sphere/disc at hit point
    [SerializeField] private TextMeshPro actionLabel;  // Shows current action

    // Networked pointer data
    private NetworkVariable<Vector3> pointerOrigin = new NetworkVariable<Vector3>(
        Vector3.zero,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    private NetworkVariable<Vector3> pointerEnd = new NetworkVariable<Vector3>(
        Vector3.zero,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    private NetworkVariable<Color> pointerColor = new NetworkVariable<Color>(
        Color.cyan,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    private NetworkVariable<bool> isPointerActive = new NetworkVariable<bool>(
        false,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    private NetworkVariable<FixedString32Bytes> currentAction = new NetworkVariable<FixedString32Bytes>(
        new FixedString32Bytes(""),
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Subscribe to network variable changes
        pointerOrigin.OnValueChanged += OnPointerOriginChanged;
        pointerEnd.OnValueChanged += OnPointerEndChanged;
        pointerColor.OnValueChanged += OnPointerColorChanged;
        isPointerActive.OnValueChanged += OnPointerActiveChanged;
        currentAction.OnValueChanged += OnActionChanged;

        // Initial setup
        if (pointerRay != null)
        {
            pointerRay.startWidth = pointerWidth;
            pointerRay.endWidth = pointerWidth * 0.5f;
            pointerRay.positionCount = 2;
        }

        UpdateVisuals();
    }

    public override void OnNetworkDespawn()
    {
        // Unsubscribe
        if (pointerOrigin != null) pointerOrigin.OnValueChanged -= OnPointerOriginChanged;
        if (pointerEnd != null) pointerEnd.OnValueChanged -= OnPointerEndChanged;
        if (pointerColor != null) pointerColor.OnValueChanged -= OnPointerColorChanged;
        if (isPointerActive != null) isPointerActive.OnValueChanged -= OnPointerActiveChanged;
        if (currentAction != null) currentAction.OnValueChanged -= OnActionChanged;

        base.OnNetworkDespawn();
    }

    /// <summary>
    /// Update pointer (call from controller script each frame)
    /// </summary>
    public void UpdatePointer(Vector3 origin, Vector3 end, Color color, string action = "")
    {
        if (!IsOwner) return;

        pointerOrigin.Value = origin;
        pointerEnd.Value = end;
        pointerColor.Value = color;
        isPointerActive.Value = true;

        if (!string.IsNullOrEmpty(action))
        {
            currentAction.Value = new FixedString32Bytes(action);
        }
    }

    /// <summary>
    /// Hide pointer
    /// </summary>
    public void HidePointer()
    {
        if (!IsOwner) return;
        isPointerActive.Value = false;
    }

    // Network variable change callbacks
    private void OnPointerOriginChanged(Vector3 prev, Vector3 current)
    {
        UpdateVisuals();
    }

    private void OnPointerEndChanged(Vector3 prev, Vector3 current)
    {
        UpdateVisuals();
    }

    private void OnPointerColorChanged(Color prev, Color current)
    {
        UpdateVisuals();
    }

    private void OnPointerActiveChanged(bool prev, bool current)
    {
        UpdateVisuals();
    }

    private void OnActionChanged(FixedString32Bytes prev, FixedString32Bytes current)
    {
        UpdateVisuals();
    }

    /// <summary>
    /// Update visual representation
    /// </summary>
    private void UpdateVisuals()
    {
        bool shouldShow = isPointerActive.Value && (alwaysVisible || IsOwner);

        // Update ray
        if (pointerRay != null)
        {
            pointerRay.enabled = shouldShow;

            if (shouldShow)
            {
                pointerRay.SetPosition(0, pointerOrigin.Value);
                pointerRay.SetPosition(1, pointerEnd.Value);
                pointerRay.startColor = pointerColor.Value;
                pointerRay.endColor = pointerColor.Value;
            }
        }

        // Update hit indicator
        if (hitIndicator != null)
        {
            hitIndicator.SetActive(shouldShow);

            if (shouldShow)
            {
                hitIndicator.transform.position = pointerEnd.Value;

                // Color the indicator to match pointer
                Renderer renderer = hitIndicator.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = pointerColor.Value;
                }
            }
        }

        // Update action label
        if (actionLabel != null)
        {
            actionLabel.gameObject.SetActive(shouldShow && !IsOwner && currentAction.Value.Length > 0);

            if (shouldShow && !IsOwner)
            {
                actionLabel.text = $"Teacher: {currentAction.Value}";
                actionLabel.transform.position = pointerEnd.Value + Vector3.up * 0.2f;

                // Billboard to face camera
                Camera mainCam = Camera.main;
                if (mainCam != null)
                {
                    actionLabel.transform.LookAt(mainCam.transform);
                    actionLabel.transform.Rotate(0, 180, 0);
                }
            }
        }
    }

    /// <summary>
    /// Get action string based on pointer color
    /// </summary>
    public static string GetActionFromColor(Color color)
    {
        if (color == Color.red) return "Annotating";
        if (color == Color.green) return "Panning";
        if (color == Color.blue) return "Zooming";
        if (color == Color.yellow) return "Rotating";
        if (color == Color.magenta) return "Reanchoring";
        return "Pointing";
    }

    // Public properties
    public Vector3 PointerOrigin => pointerOrigin.Value;
    public Vector3 PointerEnd => pointerEnd.Value;
    public Color PointerColor => pointerColor.Value;
    public bool IsActive => isPointerActive.Value;
    public string CurrentAction => currentAction.Value.ToString();
}

/*
 * STUDENT-VISIBLE POINTER SYSTEM:
 * ================================
 *
 * Purpose: Let students see where teacher is pointing and what they're doing
 *
 * Features:
 * - Colored ray from teacher's controller
 * - Hit indicator (sphere/disc) at pointed location
 * - Action label: "Teacher: Zooming" (floating text)
 * - Real-time synchronization (<50ms)
 *
 * Setup:
 * ======
 *
 * 1. Add to Teacher Player Prefab:
 *    - Add NetworkedTeacherPointer component
 *    - Create child GameObject: "PointerVisuals"
 *    - Add LineRenderer to PointerVisuals
 *    - Create sphere for hit indicator
 *    - Add TextMeshPro for action label
 *
 * 2. Integration with PointerBasedTerrainController:
 *
 *    In PointerBasedTerrainController.Update():
 *
 *    if (networkedPointer != null)
 *    {
 *        Vector3 rayOrigin = controllerPos;
 *        Vector3 rayEnd = lastHitValid ? lastHitPoint : controllerPos + rayDir * 10f;
 *        Color rayColor = GetPointerColor();
 *        string action = GetCurrentActionString();
 *
 *        networkedPointer.UpdatePointer(rayOrigin, rayEnd, rayColor, action);
 *    }
 *
 * 3. Visual Feedback for Students:
 *    - Students see teacher's pointer ray (color-coded)
 *    - Cyan = Teacher pointing (idle)
 *    - Blue = Teacher zooming
 *    - Green = Teacher panning
 *    - Yellow = Teacher rotating
 *    - Red = Teacher placing marker
 *    - White→Magenta = Teacher reanchoring
 *
 * 4. Action Labels:
 *    - "Teacher: Zooming"
 *    - "Teacher: Panning"
 *    - "Teacher: Rotating"
 *    - "Teacher: Placing Marker"
 *    - "Teacher: Reanchoring"
 *
 * Benefits:
 * =========
 * - Students know what teacher is doing
 * - Students can anticipate terrain movements
 * - Clear visual communication
 * - Better learning experience
 * - Students can point out features to each other
 *   by referencing teacher's pointer
 */