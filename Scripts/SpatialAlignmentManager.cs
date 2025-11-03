using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Manages spatial alignment for same-room multi-user scenarios
/// Supports TWO methods:
/// 1. Manual Alignment - Quick reference-based (10 seconds setup)
/// 2. Shared Spatial Anchors - Robust Meta anchors (one-time 2-3 min setup)
///
/// Use for: Teacher and students at SAME physical table
/// Not needed for: Remote students at different tables
/// </summary>
public class SpatialAlignmentManager : NetworkBehaviour
{
    public enum AlignmentMode
    {
        None,              // No alignment (for remote users)
        Manual,            // Quick reference-based alignment
        SharedAnchor       // Meta Spatial Anchors (robust)
    }

    [Header("Alignment Settings")]
    [SerializeField] private AlignmentMode alignmentMode = AlignmentMode.Manual;
    [SerializeField] private bool enableAlignment = true;

    [Header("References")]
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private ManualAlignmentCalibrator manualCalibrator;
    [SerializeField] private SharedAnchorAlignmentSystem anchorSystem;

    [Header("Debug")]
    [SerializeField] private bool showAlignmentGizmos = true;
    [SerializeField] private GameObject alignmentVisualizer;

    // Networked alignment data
    private NetworkVariable<Quaternion> worldRotationOffset = new NetworkVariable<Quaternion>(
        Quaternion.identity,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<bool> isAligned = new NetworkVariable<bool>(
        false,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<int> currentAlignmentMode = new NetworkVariable<int>(
        (int)AlignmentMode.None,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Subscribe to alignment changes
        worldRotationOffset.OnValueChanged += OnRotationOffsetChanged;
        isAligned.OnValueChanged += OnAlignmentStateChanged;

        // Auto-find calibrator components if not assigned
        if (manualCalibrator == null)
        {
            manualCalibrator = GetComponent<ManualAlignmentCalibrator>();
        }

        if (anchorSystem == null)
        {
            anchorSystem = GetComponent<SharedAnchorAlignmentSystem>();
        }
    }

    public override void OnNetworkDespawn()
    {
        if (worldRotationOffset != null) worldRotationOffset.OnValueChanged -= OnRotationOffsetChanged;
        if (isAligned != null) isAligned.OnValueChanged -= OnAlignmentStateChanged;

        base.OnNetworkDespawn();
    }

    private void Update()
    {
        // Update visualizations
        if (showAlignmentGizmos && alignmentVisualizer != null)
        {
            alignmentVisualizer.SetActive(isAligned.Value);
        }
    }

    /// <summary>
    /// Start alignment process based on selected mode
    /// </summary>
    public void StartAlignment()
    {
        if (!enableAlignment)
        {
            Debug.Log("Spatial alignment disabled");
            return;
        }

        switch (alignmentMode)
        {
            case AlignmentMode.Manual:
                StartManualAlignment();
                break;

            case AlignmentMode.SharedAnchor:
                StartSharedAnchorAlignment();
                break;

            case AlignmentMode.None:
                Debug.Log("No alignment mode selected");
                break;
        }
    }

    /// <summary>
    /// Start manual reference-based alignment
    /// </summary>
    private void StartManualAlignment()
    {
        if (manualCalibrator == null)
        {
            Debug.LogError("ManualAlignmentCalibrator not found!");
            return;
        }

        manualCalibrator.StartCalibration();
        Debug.Log("Starting manual alignment - Point at reference object");
    }

    /// <summary>
    /// Start shared spatial anchor alignment
    /// </summary>
    private void StartSharedAnchorAlignment()
    {
        if (anchorSystem == null)
        {
            Debug.LogError("SharedAnchorAlignmentSystem not found!");
            return;
        }

        if (IsServer)
        {
            // Teacher creates anchor
            anchorSystem.CreateSharedAnchor();
        }
        else
        {
            // Students locate anchor
            anchorSystem.LocateSharedAnchor();
        }
    }

    /// <summary>
    /// Set rotation offset from teacher's forward (called by calibration systems)
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void SetRotationOffsetServerRpc(Quaternion offset)
    {
        if (!IsServer) return;

        worldRotationOffset.Value = offset;
        isAligned.Value = true;
        currentAlignmentMode.Value = (int)alignmentMode;

        Debug.Log($"Alignment set: Mode={alignmentMode}, Offset={offset.eulerAngles}");
    }

    /// <summary>
    /// Apply rotation offset to client's local terrain view
    /// </summary>
    private void OnRotationOffsetChanged(Quaternion previousValue, Quaternion newValue)
    {
        if (terrainTransform != null && !IsServer)
        {
            // Apply rotation offset to student's view
            terrainTransform.rotation = newValue * terrainTransform.rotation;
            Debug.Log($"Applied alignment rotation: {newValue.eulerAngles}");
        }
    }

    private void OnAlignmentStateChanged(bool previousValue, bool newValue)
    {
        Debug.Log($"Alignment state: {(newValue ? "Aligned" : "Not Aligned")}");
    }

    /// <summary>
    /// Reset alignment (remove spatial correction)
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void ResetAlignmentServerRpc()
    {
        if (!IsServer) return;

        worldRotationOffset.Value = Quaternion.identity;
        isAligned.Value = false;

        Debug.Log("Alignment reset");
    }

    /// <summary>
    /// Change alignment mode (can be toggled in UI)
    /// </summary>
    public void SetAlignmentMode(AlignmentMode mode)
    {
        alignmentMode = mode;
        Debug.Log($"Alignment mode changed to: {mode}");
    }

    // Public properties
    public bool IsAligned => isAligned.Value;
    public AlignmentMode CurrentMode => (AlignmentMode)currentAlignmentMode.Value;
    public Quaternion RotationOffset => worldRotationOffset.Value;
    public bool AlignmentEnabled => enableAlignment;
}

/*
 * SPATIAL ALIGNMENT GUIDE:
 * ========================
 *
 * WHEN TO USE:
 * - Teacher and students at SAME physical table
 * - Need synchronized viewpoints
 * - "Teacher sees north, student at east sees correct east view"
 *
 * WHEN NOT NEEDED:
 * - Remote students (different physical tables)
 * - Each student at own desk
 * - Current system already works for this!
 *
 * METHOD 1: MANUAL ALIGNMENT (Recommended for Quick Setup)
 * =========================================================
 *
 * Setup Time: 10 seconds
 * Robustness: Good (±5-10cm, ±2-3°)
 * Persistence: Must redo each session
 *
 * Workflow:
 * 1. Teacher points at reference (e.g., classroom door, whiteboard)
 * 2. Students point at same reference
 * 3. System calculates rotation offsets
 * 4. Applies correction to each student's view
 *
 * METHOD 2: SHARED SPATIAL ANCHORS (For Permanent Setup)
 * =======================================================
 *
 * Setup Time: 2-3 minutes (one-time)
 * Robustness: Excellent (±1cm, ±0.5°)
 * Persistence: Saved, reusable across sessions
 *
 * Workflow:
 * 1. Teacher places anchor at table corner (one-time)
 * 2. Anchor saved to Meta cloud
 * 3. Students locate same anchor
 * 4. Perfect alignment automatically
 * 5. Works for weeks/months
 *
 * CHOOSING WHICH METHOD:
 * ======================
 *
 * Use Manual Alignment if:
 * - Quick demos (30-45 min)
 * - Different rooms each day
 * - Furniture moves around
 * - Simple setup preferred
 *
 * Use Shared Anchors if:
 * - Dedicated MR classroom
 * - Daily/weekly use
 * - Permanent table setup
 * - Maximum precision needed
 * - Long sessions (1+ hours)
 *
 * CONFIGURATION:
 * ==============
 *
 * In Unity Inspector (SpatialAlignmentManager):
 * - Alignment Mode: Manual or SharedAnchor
 * - Enable Alignment: true (for same room), false (for remote)
 *
 * For same-room classroom: Enable = true, Mode = Manual
 * For remote students: Enable = false
 * For permanent classroom: Enable = true, Mode = SharedAnchor
 */