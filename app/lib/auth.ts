import { NextAuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import bcrypt from 'bcryptjs';

// Simple hardcoded test user - replace with database later
const testUser = {
  id: '1',
  email: 'admin@test.com',
  password: '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/lewmyBSWKl2/Gl4yq', // 'admin123'
  name: 'Admin User',
  role: 'admin'
};

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        // Simple test user authentication
        if (credentials.email === testUser.email) {
          const isPasswordValid = await bcrypt.compare(
            credentials.password,
            testUser.password
          );

          if (isPasswordValid) {
            return {
              id: testUser.id,
              email: testUser.email,
              name: testUser.name,
              role: testUser.role,
            };
          }
        }

        return null;
      }
    })
  ],
  session: {
    strategy: 'jwt',
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role;
      }
      return token;
    },
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.sub!;
        session.user.role = token.role;
      }
      return session;
    },
  },
  pages: {
    signIn: '/login',
  },
};