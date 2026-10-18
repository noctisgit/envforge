import type { NextConfig } from "next";

// All browser calls go to /api/* on this origin and are proxied to FastAPI,
// so the session cookie is first-party and no CORS is involved.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
