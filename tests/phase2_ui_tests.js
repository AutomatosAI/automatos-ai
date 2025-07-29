const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function runPhase2UITests() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  const results = {
    timestamp: new Date().toISOString(),
    tests: [],
    screenshots: []
  };

  try {
    // Navigate to the application
    console.log('🚀 Starting Phase 2 UI Testing...');
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Take initial screenshot
    const screenshotPath = `screenshots/dashboard_initial_${Date.now()}.png`;
    await page.screenshot({ path: screenshotPath, fullPage: true });
    results.screenshots.push(screenshotPath);
    
    // Test 1: Agent Management - Create Agent
    console.log('📝 Testing Agent Management - Create Agent');
    await page.click('text=Agent Management');
    await page.waitForTimeout(2000);
    
    const agentScreenshot = `screenshots/agent_management_${Date.now()}.png`;
    await page.screenshot({ path: agentScreenshot, fullPage: true });
    results.screenshots.push(agentScreenshot);
    
    // Try to create a new agent
    try {
      await page.click('text=Create Agent', { timeout: 5000 });
      await page.waitForTimeout(2000);
      
      const createAgentScreenshot = `screenshots/create_agent_${Date.now()}.png`;
      await page.screenshot({ path: createAgentScreenshot, fullPage: true });
      results.screenshots.push(createAgentScreenshot);
      
      results.tests.push({
        name: 'Agent Management - Create Agent UI',
        status: 'PASS',
        details: 'Successfully navigated to agent creation interface'
      });
    } catch (error) {
      results.tests.push({
        name: 'Agent Management - Create Agent UI',
        status: 'FAIL',
        details: `Error: ${error.message}`
      });
    }
    
    // Test 2: Document Management
    console.log('📄 Testing Document Management');
    await page.click('text=Document Management');
    await page.waitForTimeout(2000);
    
    const docScreenshot = `screenshots/document_management_${Date.now()}.png`;
    await page.screenshot({ path: docScreenshot, fullPage: true });
    results.screenshots.push(docScreenshot);
    
    results.tests.push({
      name: 'Document Management Navigation',
      status: 'PASS',
      details: 'Successfully navigated to document management section'
    });
    
    // Test 3: Workflow Management
    console.log('⚙️ Testing Workflow Management');
    await page.click('text=Workflow Management');
    await page.waitForTimeout(2000);
    
    const workflowScreenshot = `screenshots/workflow_management_${Date.now()}.png`;
    await page.screenshot({ path: workflowScreenshot, fullPage: true });
    results.screenshots.push(workflowScreenshot);
    
    results.tests.push({
      name: 'Workflow Management Navigation',
      status: 'PASS',
      details: 'Successfully navigated to workflow management section'
    });
    
    // Test 4: System Settings
    console.log('⚙️ Testing System Settings');
    await page.click('text=Settings');
    await page.waitForTimeout(2000);
    
    const settingsScreenshot = `screenshots/system_settings_${Date.now()}.png`;
    await page.screenshot({ path: settingsScreenshot, fullPage: true });
    results.screenshots.push(settingsScreenshot);
    
    results.tests.push({
      name: 'System Settings Navigation',
      status: 'PASS',
      details: 'Successfully navigated to system settings section'
    });
    
    // Test 5: Performance Analytics
    console.log('📊 Testing Performance Analytics');
    await page.click('text=Performance Analytics');
    await page.waitForTimeout(2000);
    
    const analyticsScreenshot = `screenshots/performance_analytics_${Date.now()}.png`;
    await page.screenshot({ path: analyticsScreenshot, fullPage: true });
    results.screenshots.push(analyticsScreenshot);
    
    results.tests.push({
      name: 'Performance Analytics Navigation',
      status: 'PASS',
      details: 'Successfully navigated to performance analytics section'
    });
    
    // Test 6: Context Engineering
    console.log('🧠 Testing Context Engineering');
    await page.click('text=Context Engineering');
    await page.waitForTimeout(2000);
    
    const contextScreenshot = `screenshots/context_engineering_${Date.now()}.png`;
    await page.screenshot({ path: contextScreenshot, fullPage: true });
    results.screenshots.push(contextScreenshot);
    
    results.tests.push({
      name: 'Context Engineering Navigation',
      status: 'PASS',
      details: 'Successfully navigated to context engineering section'
    });
    
    console.log('✅ Phase 2 UI Testing completed successfully!');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
    results.tests.push({
      name: 'General UI Testing',
      status: 'FAIL',
      details: `Unexpected error: ${error.message}`
    });
  } finally {
    await browser.close();
  }
  
  // Save results
  fs.writeFileSync('phase2_ui_test_results.json', JSON.stringify(results, null, 2));
  console.log('📊 Test results saved to phase2_ui_test_results.json');
  
  return results;
}

// Run the tests
runPhase2UITests().catch(console.error);
