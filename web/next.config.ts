import type { NextConfig } from "next";

// All browser calls go to /api/* on this origin and are proxied to FastAPI,
// so the session cookie is first-party and no CORS is involved.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  // the /api proxy buffers request bodies and cuts them at 10 MB by default; builder uploads
  // send 32 MB chunks (and up to 95 MB in one request), so allow what the API itself allows
  experimental: { middlewareClientMaxBodySize: "100mb" },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
