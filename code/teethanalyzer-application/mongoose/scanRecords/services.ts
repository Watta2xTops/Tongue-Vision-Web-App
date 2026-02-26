import ScanRecord from "./model";
import { ScanRecordType } from "./schema";
import User from "../users/model";

// Create a new scan record
export async function createScanRecord(recordData: Partial<ScanRecordType>) {
  const newRecord = await ScanRecord.create(recordData);

  await User.findByIdAndUpdate(
    recordData.user,
    { $push: { scanRecords: newRecord._id } },
    { new: true }
  );

  return newRecord;
}

// Get scan record by ID
export async function findScanRecordById(recordId: string) {
  return await ScanRecord.findById(recordId).populate("user");
}

// Get all scan records for a user
export async function findScanRecordsByUser(userId: string) {
  return await ScanRecord.find({ user: userId });
}

// Update a scan record
export async function updateScanRecord(recordId: string, update: Partial<ScanRecordType>) {
  return await ScanRecord.findByIdAndUpdate(recordId, update, { new: true });
}

// Delete a scan record
export async function deleteScanRecord(recordId: string) {
  return await ScanRecord.findByIdAndDelete(recordId);
}
