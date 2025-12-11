import { betterAuth } from "better-auth";

export const auth = betterAuth({
  database: {
    provider: "postgres",
    url: process.env.DATABASE_URL || "",
  },
  emailAndPassword: {
    enabled: true,
  },
  user: {
    additionalFields: {
      softwareBackground: {
        type: "string",
        required: false,
      },
      hardwareBackground: {
        type: "string",
        required: false,
      },
      experienceLevel: {
        type: "string",
        required: false,
      },
    },
  },
});