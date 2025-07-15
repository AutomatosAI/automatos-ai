
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  // Create admin user with email and hashed password
  const hashedPassword = await bcrypt.hash('admin123', 10);
  
  const adminUser = await prisma.user.upsert({
    where: { email: 'admin@xplaincrypto.ai' },
    update: {},
    create: {
      email: 'admin@xplaincrypto.ai',
      name: 'Administrator',
      password: hashedPassword,
      role: 'admin',
    },
  });

  // Create test user for testing
  const testPassword = await bcrypt.hash('johndoe123', 10);
  
  const testUser = await prisma.user.upsert({
    where: { email: 'john@doe.com' },
    update: {},
    create: {
      email: 'john@doe.com',
      name: 'John Doe',
      password: testPassword,
      role: 'user',
    },
  });

  console.log('Database seeded successfully');
  console.log('Admin user:', adminUser);
  console.log('Test user:', testUser);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
