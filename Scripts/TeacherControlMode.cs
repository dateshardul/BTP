using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Manages role-based access control for teacher vs student
/// Only teachers can manipulate the terrain and add annotations
/// Students can only view
/// </summary>
public class TeacherControlMode : NetworkBehaviour
{
    public enum UserRole
    {
        Student,    // Can only view
        Teacher     // Can manipulate terrain and annotate
    }

    [Header("Role Settings")]
    [SerializeField] private UserRole defaultRole = UserRole.Student;

    // Networked role for this client
    private NetworkVariable<int> userRole = new NetworkVariable<int>(
        (int)UserRole.Student,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    [Header("UI References")]
    [SerializeField] private GameObject teacherControlUI;  // Shows when teacher
    [SerializeField] private GameObject studentViewUI;     // Shows when student

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        if (IsOwner)
        {
            // Set role on spawn
            userRole.Value = (int)defaultRole;
        }

        // Subscribe to role changes
        userRole.OnValueChanged += OnRoleChanged;

        // Initial UI update
        UpdateUI();
    }

    public override void OnNetworkDespawn()
    {
        if (userRole != null)
        {
            userRole.OnValueChanged -= OnRoleChanged;
        }
        base.OnNetworkDespawn();
    }

    private void OnRoleChanged(int previousValue, int newValue)
    {
        UpdateUI();
        Debug.Log($"User role changed from {(UserRole)previousValue} to {(UserRole)newValue}");
    }

    /// <summary>
    /// Check if this user is a teacher
    /// </summary>
    public bool IsTeacher()
    {
        return (UserRole)userRole.Value == UserRole.Teacher;
    }

    /// <summary>
    /// Check if this user is a student
    /// </summary>
    public bool IsStudent()
    {
        return (UserRole)userRole.Value == UserRole.Student;
    }

    /// <summary>
    /// Set role for this user (only owner can change their own role)
    /// </summary>
    public void SetRole(UserRole newRole)
    {
        if (!IsOwner)
        {
            Debug.LogWarning("Only owner can set their own role");
            return;
        }

        userRole.Value = (int)newRole;
        Debug.Log($"Role set to: {newRole}");
    }

    /// <summary>
    /// Server RPC to promote a client to teacher
    /// Only server/host can call this
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void PromoteToTeacherServerRpc(ulong clientId)
    {
        if (!IsServer) return;

        // Find the player object for this client
        if (NetworkManager.Singleton.ConnectedClients.TryGetValue(clientId, out var client))
        {
            var teacherControl = client.PlayerObject?.GetComponent<TeacherControlMode>();
            if (teacherControl != null)
            {
                teacherControl.SetRoleClientRpc(UserRole.Teacher);
            }
        }
    }

    /// <summary>
    /// Client RPC to set role (server calls this)
    /// </summary>
    [ClientRpc]
    private void SetRoleClientRpc(UserRole newRole)
    {
        if (IsOwner)
        {
            userRole.Value = (int)newRole;
        }
    }

    /// <summary>
    /// Update UI based on current role
    /// </summary>
    private void UpdateUI()
    {
        bool isTeacher = IsTeacher();

        if (teacherControlUI != null)
        {
            teacherControlUI.SetActive(isTeacher && IsOwner);
        }

        if (studentViewUI != null)
        {
            studentViewUI.SetActive(!isTeacher && IsOwner);
        }
    }

    /// <summary>
    /// Get current role
    /// </summary>
    public UserRole GetRole()
    {
        return (UserRole)userRole.Value;
    }

    /// <summary>
    /// Get role as string for UI display
    /// </summary>
    public string GetRoleString()
    {
        return ((UserRole)userRole.Value).ToString();
    }

    // Public property for easy access
    public bool CanManipulateTerrain => IsTeacher() && IsOwner;
    public bool CanAnnotate => IsTeacher() && IsOwner;
}