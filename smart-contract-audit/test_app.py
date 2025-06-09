"""
Test script for Smart Contract Audit Tool
"""
import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_CONTRACT = """
pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    // Vulnerable function - no access control
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
    
    // Vulnerable function - reentrancy
    function withdrawAll() public {
        uint256 amount = balances[msg.sender];
        payable(msg.sender).call{value: amount}("");
        balances[msg.sender] = 0;
    }
    
    // Function with potential overflow (older Solidity)
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Emergency function without proper access control
    function emergencyWithdraw() public {
        payable(owner).transfer(address(this).balance);
    }
}
"""

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_audit_workflow():
    """Test complete audit workflow"""
    print("\nTesting audit workflow...")
    
    try:
        # Start audit
        print("Starting audit...")
        response = requests.post(f"{BASE_URL}/api/audit/start", 
                               json={
                                   "code_snippet": TEST_CONTRACT,
                                   "static_tool": "Test static analysis results"
                               })
        
        if response.status_code != 201:
            print(f"❌ Failed to start audit: {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        session_id = data['session_id']
        print(f"✅ Audit started with session ID: {session_id}")
        
        # Monitor progress
        print("Monitoring progress...")
        max_attempts = 30  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(f"{BASE_URL}/api/audit/{session_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                status = status_data['status']
                progress = status_data.get('progress', {})
                
                print(f"Status: {status}, Progress: {progress.get('progress_percentage', 0):.1f}%")
                
                if status == 'completed':
                    print("✅ Audit completed successfully")
                    
                    # Get results
                    response = requests.get(f"{BASE_URL}/api/audit/{session_id}/results")
                    if response.status_code == 200:
                        results = response.json()
                        print(f"✅ Results retrieved: {results.get('finding_number', 0)} findings")
                        return True
                    else:
                        print(f"❌ Failed to get results: {response.status_code}")
                        return False
                        
                elif status == 'failed':
                    print("❌ Audit failed")
                    return False
                    
            time.sleep(10)  # Wait 10 seconds
            attempt += 1
        
        print("❌ Audit timed out")
        return False
        
    except Exception as e:
        print(f"❌ Audit workflow error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\nTesting API endpoints...")
    
    # Test sessions list
    try:
        response = requests.get(f"{BASE_URL}/api/audit/sessions")
        if response.status_code == 200:
            print("✅ Sessions list endpoint working")
        else:
            print(f"❌ Sessions list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Sessions list error: {e}")

def main():
    """Run all tests"""
    print("Smart Contract Audit Tool - Test Suite")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed - make sure the server is running")
        return
    
    # Test API endpoints
    test_api_endpoints()
    
    # Test audit workflow (commented out for now as it requires API keys)
    print("\n⚠️  Audit workflow test requires AI API keys")
    print("   Set GEMINI_API_KEY or OPENAI_API_KEY in .env file to test")
    
    print("\n" + "=" * 50)
    print("Test suite completed")

if __name__ == "__main__":
    main()

