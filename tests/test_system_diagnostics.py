import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.system_diagnostics import SystemDiagnostics

class TestSystemDiagnostics(unittest.TestCase):
    """Test suite for system diagnostics functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.diagnostics = SystemDiagnostics()
    
    def test_system_health(self):
        """Test system health check functionality"""
        health = self.diagnostics.check_system_health()
        
        # Verify structure
        self.assertIsInstance(health, dict)
        self.assertIn('cpu_usage', health)
        self.assertIn('memory_usage', health)
        self.assertIn('disk_usage', health)
        self.assertIn('status', health)
        
        # Verify data types
        self.assertIsInstance(health['cpu_usage'], float)
        self.assertIsInstance(health['memory_usage'], float)
        self.assertIsInstance(health['disk_usage'], float)
        self.assertIsInstance(health['status'], str)
        
        # Verify value ranges
        self.assertGreaterEqual(health['cpu_usage'], 0)
        self.assertLessEqual(health['cpu_usage'], 100)
        self.assertGreaterEqual(health['memory_usage'], 0)
        self.assertLessEqual(health['memory_usage'], 100)
        self.assertGreaterEqual(health['disk_usage'], 0)
        self.assertLessEqual(health['disk_usage'], 100)
    
    def test_disk_health(self):
        """Test disk health check functionality"""
        disk_health = self.diagnostics.check_disk_health()
        
        # Verify structure
        self.assertIsInstance(disk_health, dict)
        
        # Check at least one disk is present
        self.assertGreater(len(disk_health), 0)
        
        # Verify disk information
        for disk, info in disk_health.items():
            self.assertIsInstance(info, dict)
            self.assertIn('total', info)
            self.assertIn('used', info)
            self.assertIn('free', info)
            self.assertIn('percent', info)
            self.assertIn('status', info)
            
            # Verify space calculations
            self.assertGreaterEqual(info['total'], info['used'])
            self.assertGreaterEqual(info['free'], 0)
            self.assertGreaterEqual(info['percent'], 0)
            self.assertLessEqual(info['percent'], 100)
    
    def test_memory_health(self):
        """Test memory health check functionality"""
        memory = self.diagnostics.check_memory_health()
        
        # Verify structure
        self.assertIsInstance(memory, dict)
        self.assertIn('total', memory)
        self.assertIn('available', memory)
        self.assertIn('used', memory)
        self.assertIn('percent', memory)
        self.assertIn('status', memory)
        
        # Verify calculations
        self.assertGreaterEqual(memory['total'], memory['used'])
        self.assertGreaterEqual(memory['available'], 0)
        self.assertGreaterEqual(memory['percent'], 0)
        self.assertLessEqual(memory['percent'], 100)
    
    def test_network_health(self):
        """Test network health check functionality"""
        network = self.diagnostics.check_network_health()
        
        # Verify structure
        self.assertIsInstance(network, dict)
        self.assertIn('interfaces', network)
        self.assertIn('connectivity', network)
        self.assertIn('status', network)
        
        # Verify interfaces
        self.assertIsInstance(network['interfaces'], dict)
        for interface, stats in network['interfaces'].items():
            self.assertIn('bytes_sent', stats)
            self.assertIn('bytes_recv', stats)
            self.assertIn('packets_sent', stats)
            self.assertIn('packets_recv', stats)
            
            # Verify non-negative values
            self.assertGreaterEqual(stats['bytes_sent'], 0)
            self.assertGreaterEqual(stats['bytes_recv'], 0)
            self.assertGreaterEqual(stats['packets_sent'], 0)
            self.assertGreaterEqual(stats['packets_recv'], 0)
    
    def test_startup_programs(self):
        """Test startup programs check functionality"""
        startup = self.diagnostics.check_startup_programs()
        
        # Verify structure
        self.assertIsInstance(startup, dict)
        self.assertIn('programs', startup)
        self.assertIn('status', startup)
        
        # Verify programs list
        self.assertIsInstance(startup['programs'], list)
        for program in startup['programs']:
            self.assertIn('name', program)
            self.assertIn('command', program)
            self.assertIn('impact', program)
            
            # Verify impact is valid
            self.assertIn(program['impact'], ['low', 'high'])
    
    def test_security_health(self):
        """Test security health check functionality"""
        security = self.diagnostics.check_security_health()
        
        # Verify structure
        self.assertIsInstance(security, dict)
        self.assertIn('antivirus_active', security)
        self.assertIn('firewall_active', security)
        self.assertIn('updates_pending', security)
        self.assertIn('status', security)
        
        # Verify boolean values
        self.assertIsInstance(security['antivirus_active'], bool)
        self.assertIsInstance(security['firewall_active'], bool)
        self.assertIsInstance(security['updates_pending'], bool)
    
    def test_system_repair(self):
        """Test system repair functionality"""
        # Test repair with sample issues
        issues = ['disk', 'memory', 'network']
        result = self.diagnostics.repair_system(issues)
        
        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('repairs', result)
        
        # Verify repairs
        self.assertIsInstance(result['repairs'], dict)
        for issue in issues:
            self.assertIn(issue, result['repairs'])
            self.assertIn(result['repairs'][issue], 
                         ['completed', 'failed', 'not_supported', 'scheduled'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
