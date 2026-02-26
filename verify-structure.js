const fs = require("fs");
const path = require("path");

const pathsToCheck = [
  ".docker/teethanalyzer-db/seed-mongodb.js",
  "code/teethanalyzer-application",
  "code/teethanalyzer-backend",
  "docker-compose.yml",
];

let allOk = true;

console.log("🔍 Verifying required project structure...\n");

pathsToCheck.forEach((relativePath) => {
  const fullPath = path.resolve(__dirname, relativePath);
  const exists = fs.existsSync(fullPath);

  if (exists) {
    console.log(`✅ Exists: ${relativePath}`);
  } else {
    console.log(`❌ Missing: ${relativePath}`);
    allOk = false;
  }
});

console.log("\n🎯 Verification " + (allOk ? "PASSED ✅" : "FAILED ❌"));
process.exit(allOk ? 0 : 1);
