using UnityEngine;
using Unity.Netcode;
using TMPro;

public class UserActionIndicator : NetworkBehaviour
{
    [SerializeField] private LineRenderer handIndicator;
    [SerializeField] private GameObject zoomIcon;
    [SerializeField] private GameObject panIcon;
    [SerializeField] private GameObject rotateIcon;
    [SerializeField] private TextMeshPro actionText;

    // Networked properties (adapted from Fusion [Networked])
    private NetworkVariable<Vector3> networkedHandPosition = new NetworkVariable<Vector3>(Vector3.zero);
    private NetworkVariable<int> actionType = new NetworkVariable<int>(0); // 0=none, 1=zoom, 2=pan, 3=rotate

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Subscribe to value changes
        networkedHandPosition.OnValueChanged += OnHandPositionChanged;
        actionType.OnValueChanged += OnActionTypeChanged;

        // Initial update
        UpdateIndicators();
    }

    public override void OnNetworkDespawn()
    {
        // Unsubscribe from value changes
        if (networkedHandPosition != null) networkedHandPosition.OnValueChanged -= OnHandPositionChanged;
        if (actionType != null) actionType.OnValueChanged -= OnActionTypeChanged;

        base.OnNetworkDespawn();
    }

    private void OnHandPositionChanged(Vector3 previousValue, Vector3 newValue)
    {
        UpdateIndicators();
    }

    private void OnActionTypeChanged(int previousValue, int newValue)
    {
        UpdateIndicators();
    }

    private void UpdateIndicators()
    {
        if (handIndicator == null) return;

        // Show indicator at networked hand position
        if (actionType.Value > 0)
        {
            handIndicator.SetPosition(0, networkedHandPosition.Value);
            handIndicator.SetPosition(1, networkedHandPosition.Value + Vector3.up * 0.1f);

            // Show appropriate icon
            if (zoomIcon != null) zoomIcon.SetActive(actionType.Value == 1);
            if (panIcon != null) panIcon.SetActive(actionType.Value == 2);
            if (rotateIcon != null) rotateIcon.SetActive(actionType.Value == 3);

            // Update text
            if (actionText != null)
            {
                string actionName = actionType.Value == 1 ? "Zooming" :
                                   actionType.Value == 2 ? "Panning" : "Rotating";
                actionText.text = $"User {OwnerClientId} is {actionName}";
            }
        }

        handIndicator.gameObject.SetActive(actionType.Value > 0);
    }

    // Method to update hand position from owner client
    public void UpdateHandPosition(Vector3 handPos)
    {
        if (!IsOwner) return;
        UpdateHandPositionServerRpc(handPos);
    }

    // Method to update action type from owner client
    public void UpdateActionType(int newActionType)
    {
        if (!IsOwner) return;
        UpdateActionTypeServerRpc(newActionType);
    }

    [ServerRpc(RequireOwnership = true)]
    private void UpdateHandPositionServerRpc(Vector3 handPos)
    {
        networkedHandPosition.Value = handPos;
    }

    [ServerRpc(RequireOwnership = true)]
    private void UpdateActionTypeServerRpc(int newActionType)
    {
        actionType.Value = newActionType;
    }

    // Public getters
    public Vector3 NetworkedHandPosition => networkedHandPosition.Value;
    public int ActionType => actionType.Value;
}