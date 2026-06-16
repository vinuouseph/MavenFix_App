import { NextRequest } from 'next/server';

// Disable Next.js response buffering — critical for SSE
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001';

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  const projectId = params.id;
  const backendUrl = `${BACKEND_URL}/git/stream/${projectId}`;

  // Fetch the SSE stream from the backend
  const backendRes = await fetch(backendUrl, {
    headers: {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
    },
    // @ts-expect-error – Node fetch supports duplex
    duplex: 'half',
  });

  if (!backendRes.ok || !backendRes.body) {
    return new Response('Failed to connect to backend stream', {
      status: backendRes.status,
    });
  }

  // Pipe the body directly through — no buffering
  return new Response(backendRes.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no', // disable nginx buffering
    },
  });
}
