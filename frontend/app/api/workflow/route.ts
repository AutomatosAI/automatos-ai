import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate input
      return NextResponse.json(
        { error: 'GitHub URL is required' },
        { status: 400 }
      );
    }

    // Call the actual orchestrator API
    try {
      const backendPayload = {
        repository_url: body.githubUrl,
        task_prompt: body.prompt || undefined,
      };

      const backendResponse = await fetch('http://enhanced_mcp_bridge:8001/workflow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'xplaincrypto-api-key',
        },
        body: JSON.stringify(backendPayload),
      });

        const errorText = await backendResponse.text();
        console.error('Backend API error:', errorText);
        
        return NextResponse.json(
          { error: 'Backend error: ' + errorText },
          { status: 500 }
        );
      }

      const backendResult = await backendResponse.json();
      
      return NextResponse.json({
        success: true,
        data: {
          workflowId: 'temp-' + Date.now(),
          status: 'running',
          backendWorkflowId: backendResult.workflow_id,
        },
      });

    } catch (backendError) {
      console.error('Failed to call backend API:', backendError);
      
      return NextResponse.json(
        { error: 'Failed to start workflow on backend' },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error('Error starting workflow:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  // Return empty workflows for now
  return NextResponse.json({
    success: true,
    data: [],
  });
}
