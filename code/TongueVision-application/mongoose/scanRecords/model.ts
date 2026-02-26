import mongoose, { model } from "mongoose";
import { ScanRecordSchema, ScanRecordType } from "@/mongoose/scanRecords/schema";

// "ScanRecord" is the model name (capitalized), maps to "scanrecords" collection
const ScanRecordModel = mongoose.models.ScanRecord || model<ScanRecordType>("ScanRecord", ScanRecordSchema);
export default ScanRecordModel;
