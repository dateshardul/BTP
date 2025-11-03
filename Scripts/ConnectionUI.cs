using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Simple UI for connecting to network session
/// Provides buttons for Host/Client/Server and input fields for IP/Port
/// </summary>
public class ConnectionUI : MonoBehaviour
{
    [Header("UI References")]
    [SerializeField] private Button hostButton;
    [SerializeField] private Button clientButton;
    [SerializeField] private Button serverButton;
    [SerializeField] private Button disconnectButton;
    [SerializeField] private TMP_InputField ipInputField;
    [SerializeField] private TMP_InputField portInputField;
    [SerializeField] private TextMeshProUGUI statusText;

    [Header("Manager Reference")]
    [SerializeField] private NetworkConnectionManager connectionManager;

    [Header("Default Values")]
    [SerializeField] private string defaultIP = "127.0.0.1";
    [SerializeField] private string defaultPort = "7777";

    private void Start()
    {
        // Set up button listeners
        if (hostButton != null)
            hostButton.onClick.AddListener(OnHostButtonClicked);

        if (clientButton != null)
            clientButton.onClick.AddListener(OnClientButtonClicked);

        if (serverButton != null)
            serverButton.onClick.AddListener(OnServerButtonClicked);

        if (disconnectButton != null)
        {
            disconnectButton.onClick.AddListener(OnDisconnectButtonClicked);
            disconnectButton.gameObject.SetActive(false); // Hide initially
        }

        // Set up input field listeners
        if (ipInputField != null)
        {
            ipInputField.text = defaultIP;
            ipInputField.onEndEdit.AddListener(OnIPChanged);
        }

        if (portInputField != null)
        {
            portInputField.text = defaultPort;
            portInputField.onEndEdit.AddListener(OnPortChanged);
        }

        // Find connection manager if not assigned
        if (connectionManager == null)
        {
            connectionManager = FindObjectOfType<NetworkConnectionManager>();
            if (connectionManager == null)
            {
                Debug.LogError("NetworkConnectionManager not found! Please assign it in the inspector.");
            }
        }

        UpdateStatusText("Ready to connect");
    }

    private void OnDestroy()
    {
        // Remove listeners
        if (hostButton != null)
            hostButton.onClick.RemoveListener(OnHostButtonClicked);

        if (clientButton != null)
            clientButton.onClick.RemoveListener(OnClientButtonClicked);

        if (serverButton != null)
            serverButton.onClick.RemoveListener(OnServerButtonClicked);

        if (disconnectButton != null)
            disconnectButton.onClick.RemoveListener(OnDisconnectButtonClicked);

        if (ipInputField != null)
            ipInputField.onEndEdit.RemoveListener(OnIPChanged);

        if (portInputField != null)
            portInputField.onEndEdit.RemoveListener(OnPortChanged);
    }

    private void OnHostButtonClicked()
    {
        if (connectionManager == null)
        {
            UpdateStatusText("Error: Connection Manager not found");
            return;
        }

        UpdateStatusText("Starting as Host...");
        connectionManager.StartHost();

        ShowDisconnectButton();
        HideConnectionButtons();
    }

    private void OnClientButtonClicked()
    {
        if (connectionManager == null)
        {
            UpdateStatusText("Error: Connection Manager not found");
            return;
        }

        UpdateStatusText($"Connecting to {ipInputField.text}:{portInputField.text}...");
        connectionManager.StartClient();

        ShowDisconnectButton();
        HideConnectionButtons();
    }

    private void OnServerButtonClicked()
    {
        if (connectionManager == null)
        {
            UpdateStatusText("Error: Connection Manager not found");
            return;
        }

        UpdateStatusText("Starting as Server...");
        connectionManager.StartServer();

        ShowDisconnectButton();
        HideConnectionButtons();
    }

    private void OnDisconnectButtonClicked()
    {
        if (connectionManager == null)
        {
            UpdateStatusText("Error: Connection Manager not found");
            return;
        }

        UpdateStatusText("Disconnecting...");
        connectionManager.Disconnect();

        HideDisconnectButton();
        ShowConnectionButtons();
        UpdateStatusText("Disconnected - Ready to connect");
    }

    private void OnIPChanged(string newIP)
    {
        if (connectionManager != null)
        {
            connectionManager.SetIPAddress(newIP);
        }
    }

    private void OnPortChanged(string newPort)
    {
        if (ushort.TryParse(newPort, out ushort portValue))
        {
            if (connectionManager != null)
            {
                connectionManager.SetPort(portValue);
            }
        }
        else
        {
            Debug.LogWarning($"Invalid port: {newPort}. Using default.");
            if (portInputField != null)
            {
                portInputField.text = defaultPort;
            }
        }
    }

    private void UpdateStatusText(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
        Debug.Log($"[ConnectionUI] {message}");
    }

    private void ShowDisconnectButton()
    {
        if (disconnectButton != null)
        {
            disconnectButton.gameObject.SetActive(true);
        }
    }

    private void HideDisconnectButton()
    {
        if (disconnectButton != null)
        {
            disconnectButton.gameObject.SetActive(false);
        }
    }

    private void ShowConnectionButtons()
    {
        if (hostButton != null) hostButton.gameObject.SetActive(true);
        if (clientButton != null) clientButton.gameObject.SetActive(true);
        if (serverButton != null) serverButton.gameObject.SetActive(true);
        if (ipInputField != null) ipInputField.gameObject.SetActive(true);
        if (portInputField != null) portInputField.gameObject.SetActive(true);
    }

    private void HideConnectionButtons()
    {
        if (hostButton != null) hostButton.gameObject.SetActive(false);
        if (clientButton != null) clientButton.gameObject.SetActive(false);
        if (serverButton != null) serverButton.gameObject.SetActive(false);
        if (ipInputField != null) ipInputField.gameObject.SetActive(false);
        if (portInputField != null) portInputField.gameObject.SetActive(false);
    }
}