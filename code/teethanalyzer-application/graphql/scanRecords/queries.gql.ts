export default /* GraphQL */ `
    getScanRecordById(recordId: ID!): ScanRecord
    getScanRecordsByUser(userId: ID!): [ScanRecord]
`;
