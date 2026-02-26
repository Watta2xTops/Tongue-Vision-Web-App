import * as scanService from "mongoose/scanRecords/services";

export const scanRecordQueries = {
  getScanRecordById: (_: any, { recordId }: { recordId: string }) =>
    scanService.findScanRecordById(recordId),
  getScanRecordsByUser: (_: any, { userId }: { userId: string }) =>
    scanService.findScanRecordsByUser(userId),
};
