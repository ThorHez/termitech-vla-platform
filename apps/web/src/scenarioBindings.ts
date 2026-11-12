// These records are associated through verified training input and output lineage.
export const sceneBindings: Partial<
  Record<string, { runIds?: string[]; datasetIds?: string[]; modelIds?: string[] }>
> = {
  components: {
    runIds: [
      'a0fa3bb6-adff-4583-a9b5-15528287f048',
      '922e72f5-b075-427a-ad65-51ebd0780275',
      '59d25a86-2e43-4938-aade-1602b240bc6b',
    ],
    datasetIds: [
      'd72a87d1-6d00-4fb1-a348-1f4ee5213ab9',
      '56364234-a2c5-4494-8ad9-838b18c72a82',
      'dcb06fe0-7e64-4c8e-9de8-5d446a233a01',
      '853f6c25-dd56-4bfd-b4c6-8a8a9eed7769',
      '956b9425-c499-4e5a-9802-3f56d33c4e7e',
      '82cfc6cb-064f-419f-8fd2-748bb7264f0a',
      '1cfacf7b-4bbb-4548-a5b1-37a55dbc70ce',
      'f1d7be84-6f07-4522-873b-7c1ce2ce1840',
      'ede3143b-8377-4475-a5f0-b3d96e4895e5',
      'bbbf80bb-71d9-4770-aafc-650e4775f75f',
      '7ee4e08f-3a4e-4248-b7d8-f145907807c4',
      '0a280167-205a-4c0e-a09e-bc76a1c0e75b',
      'c05601df-f761-47ea-bd79-152f2e5903af',
      'a33eb6e0-7068-46d3-81e4-942bc009652d',
    ],
    modelIds: [
      'ebadeada-d676-4cd2-89e0-ccab524d5a20',
      'c74cd548-031c-4ec0-8d97-32d68b35c665',
      '75a6dc41-2de1-4c41-b37d-107677737376',
      '8b9abc7b-9251-4c88-b794-74a69a4ed12c',
      'ae37e04e-9fc3-4129-a04e-88e4c14feb26',
      'ac3d96d2-7d75-42f7-ab5e-00f6fd29a2c4',
    ],
  },
  fold: {
    runIds: ['c4247165-7668-4d09-afa1-c3b41dacd8df'],
    datasetIds: [
      'a3ea152e-390e-43e9-9d02-bacb39d445d1',
      '33adda09-044d-42fb-8120-d4f72c925bfb',
      'fb321df3-dd32-4bbb-8253-6dde36a3b2b3',
      'c4c799b9-90cd-4ef1-9a7a-09d90ae8a6a5',
    ],
    modelIds: ['66a75fb2-2ee4-411e-8a20-3ea905c17b72', 'b91a1a51-0ef6-4170-b6a4-825eec447830'],
  },
  goods: {
    runIds: ['d7f199e5-c649-45d8-910d-f2354fa5de9f'],
    datasetIds: ['68b25ca0-4b98-4c40-96c0-b8f46c4605a3'],
    modelIds: [
      '8baee31e-a206-4bd8-ae5c-eeed91b70be4',
      'd1c9c582-720b-47e2-9d9c-8e45360a3031',
      'eecb9d46-a7ad-4e8d-8e36-ba03c814334f',
    ],
  },
};
