import * as scanService from "mongoose/scanRecords/services";

export const scanRecordMutations = {
  createScanRecord: (_: any, args: any) => scanService.createScanRecord(args),
  updateScanRecord: (_: any, { recordId, ...update }: any) =>
    scanService.updateScanRecord(recordId, update),
  deleteScanRecord: (_: any, { recordId }: { recordId: string }) =>
    scanService.deleteScanRecord(recordId),
};
