
import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { PrismaClient } from '@prisma/client';
import { WorkflowFormData } from '@/lib/types';
import { authOptions } from '@/lib/auth';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const body: WorkflowFormData = await request.json();
    
    // Validate input
    if (!body.githubUrl) {
      return NextResponse.json(
        { error: 'GitHub URL is required' },
        { status: 400 }
      );
    }

    if (!body.mode || !['self-contained', 'prompt-driven'].includes(body.mode)) {
      return NextResponse.json(
        { error: 'Valid mode is required' },
        { status: 400 }
      );
    }

    // Create workflow in database
    const workflow = await prisma.workflow.create({
      data: {
        githubUrl: body.githubUrl,
        prompt: body.prompt || null,
        mode: body.mode,
        userId: session.user.id,
        status: 'pending',
        progress: 0,
      },
    });

    // TODO: Integrate with actual orchestrator API
    // For now, simulate starting the workflow
    
    return NextResponse.json({
      success: true,
      data: {
        workflowId: workflow.id,
        status: workflow.status,
      },
    });

  } catch (error) {
    console.error('Error starting workflow:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const workflows = await prisma.workflow.findMany({
      where: {
        userId: session.user.id,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      data: workflows,
    });

  } catch (error) {
    console.error('Error fetching workflows:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
